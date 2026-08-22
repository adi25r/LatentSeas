import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import List, Dict, Tuple
from embeddings import get_embedder


class LatentExplorer:
    def __init__(self, model_name: str = "gpt2-small", device: str = "cpu"):
        """
        Initialize the model and SAE.

        Args:
            model_name: Name of the GPT-2 model to load
            device: Device 
        """
        self.device = device
        self.model_name = model_name

        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device
        )

        # TODO: make this layer a configuration, chosen by the user
        self.sae_layer = 6
        self.sae = None
        self.feature_activations = None

        print(f"Model loaded on {device}")

    def load_sae(self, sae_release: str = "gpt2-small-res-jb",
                 sae_id: str = "blocks.6.hook_resid_pre"):
        """
        Load Sparse Autoencoder for a specific layer.

        Args:
            sae_release: SAE release name
            sae_id: Specific SAE identifier for the layer
        """
        self.sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=self.device)
        print(f"SAE loaded with {self.sae.cfg.d_sae} features")

    @property
    def hook_point(self) -> str:
        return f"blocks.{self.sae_layer}.hook_resid_pre"

    def get_feature_activations(self, text: str, skip_bos: bool = True) -> torch.Tensor:
        """
        Get SAE feature activations that occur when the model processes the given text.

        The BOS token is dropped by default. Its residual stream is an outlier that makes
        the SAE fire a fixed set of huge (~400) features unrelated to the text, which would
        otherwise swamp every content feature (~20-50).

        Args:
            text: Input text to feed to the model (the stimulus that triggers activations)
            skip_bos: Drop position 0 from the returned activations

        Returns:
            Tensor of feature activations [seq_len, d_sae]
        """
        if self.sae is None:
            raise ValueError("SAE not loaded. Call load_sae() first.")

        # forward pass for activations
        tokens = self.model.to_tokens(text)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=[self.hook_point])
            activations = cache[self.hook_point]  # [batch, seq_len, d_model]
            feature_acts = self.sae.encode(activations)  # [batch, seq_len, d_sae]

        feature_acts = feature_acts[0]  # Remove batch dimension
        if skip_bos:
            feature_acts = feature_acts[1:]

        self.feature_activations = feature_acts
        return self.feature_activations

    def print_top_features(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Print the top-k most active features for the given text.

        Args:
            text: Input text to analyze
            top_k: Number of top features to show

        Returns:
            List of (feature_idx, activation_strength) tuples
        """
        feature_acts = self.get_feature_activations(text)

        # Get max activation across sequence for each feature
        max_acts = feature_acts.max(dim=0).values  # [d_sae]

        # Get top-k features
        top_values, top_indices = torch.topk(max_acts, top_k)

        print(f"\nTop {top_k} features for text: '{text}'")
        print("-" * 60)

        results = []
        for idx, (feat_idx, activation) in enumerate(zip(top_indices, top_values)):
            feat_idx = feat_idx.item()
            activation = activation.item()
            print(f"{idx+1:2d}. Feature {feat_idx:5d}: {activation:.4f}")
            results.append((feat_idx, activation))

        return results

    def steering_vector(self, boosted_features: Dict[int, float]) -> torch.Tensor:
        """Sum of decoder directions, each scaled by its strength. W_dec rows are unit norm,
        so strength is in residual-norm units (content features naturally fire at ~20-50)."""
        steer = torch.zeros(self.sae.cfg.d_in, device=self.device)
        for feat_idx, strength in boosted_features.items():
            steer = steer + strength * self.sae.W_dec[feat_idx]
        return steer

    def generate_with_boosted_features(self, prompt: str, boosted_features: Dict[int, float],
        max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        Generate text with specific SAE features steered.

        Adds the features' decoder directions into the residual stream. It deliberately does
        not round-trip through decode(encode(x)): that discards ~24% of the residual norm and
        raises CE loss even with no features selected. Multiplying encoded activations also
        cannot work here, since a feature that is off at a position is exactly 0 and stays 0
        at any multiplier.

        Args:
            prompt: Input prompt text
            boosted_features: Dict mapping feature indices to steering strengths
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if self.sae is None:
            raise ValueError("SAE not loaded. Call load_sae() first.")

        tokens = self.model.to_tokens(prompt)
        steer = self.steering_vector(boosted_features)

        def steer_hook(resid, hook):
            # Never steer BOS - perturbing that outlier position derails generation.
            # With KV caching the prompt arrives as one pass, then one token at a time.
            if resid.shape[1] > 1:
                resid[:, 1:, :] += steer
            else:
                resid += steer
            return resid

        with self.model.hooks(fwd_hooks=[(self.hook_point, steer_hook)]):
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                verbose=False
            )

        return self.model.to_string(generated_tokens[0]).replace("<|endoftext|>", "")

    def list_features_in_range(self, start: int = 0, end: int = 100) -> None:
        """
        List features in a given range.

        Args:
            start: Start index
            end: End index
        """
        if self.sae is None:
            raise ValueError("SAE not loaded. Call load_sae() first.")

        total_features = self.sae.cfg.d_sae
        end = min(end, total_features)

        print(f"\nFeatures {start} to {end} (Total: {total_features})")
        print("-" * 60)
        for i in range(start, end):
            print(f"Feature {i:5d}")

    def get_feature_stats(self, text: str, feature_idx: int) -> Dict:
        """
        Get detailed statistics about a specific feature's activation.

        Args:
            text: Input text to analyze
            feature_idx: Feature index to analyze

        Returns:
            Dictionary with feature statistics
        """
        feature_acts = self.get_feature_activations(text)
        feature_activation = feature_acts[:, feature_idx]  # [seq_len]

        stats = {
            'feature_idx': feature_idx,
            'max': feature_activation.max().item(),
            'mean': feature_activation.mean().item(),
            'min': feature_activation.min().item(),
            'std': feature_activation.std().item(),
            'activations': feature_activation.cpu().numpy()
        }

        return stats

    def get_pointmap(self, strategy: str = "umap3d", **kwargs) -> np.ndarray:
        """
        Compute 3D embedding of all SAE features.

        Args:
            strategy: Embedding strategy - 'umap3d', 'pca2d+norm', 'pca2d+variance', 'umap2d+norm', etc.
            **kwargs: Additional arguments for the embedder

        Returns:
            Array of shape [d_sae, 3] with (x, y, z) coordinates for each feature
        """
        if self.sae is None:
            raise ValueError("SAE not loaded. Call load_sae() first.")

        embedder = get_embedder(strategy, **kwargs)
        feature_weights = self.sae.W_dec
        pointmap = embedder.compute_embedding(feature_weights)

        return pointmap
