import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import umap


class FeatureEmbedder(ABC):
    @abstractmethod
    def compute_embedding(self, feature_weights: torch.Tensor) -> np.ndarray:
        """Return [d_sae, 3] array of (x, y, z) coordinates for each feature."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class UMAP3DEmbedder(FeatureEmbedder):
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def compute_embedding(self, feature_weights: torch.Tensor) -> np.ndarray:
        weights_np = feature_weights.cpu().detach().numpy()
        print(f"Computing UMAP 3D embedding for {weights_np.shape[0]} features...")

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            verbose=True
        )

        embedding = reducer.fit_transform(weights_np)
        print(f"UMAP embedding complete. Shape: {embedding.shape}")
        return embedding

    def get_name(self) -> str:
        return "UMAP-3D"


class PCA2DPlusPropertyEmbedder(FeatureEmbedder):
    def __init__(self, property_type: str = "norm", random_state: int = 42):
        """property_type: 'norm', 'variance', or 'sparsity'"""
        self.property_type = property_type
        self.random_state = random_state
        self.feature_sparsity = None

    def set_sparsity_stats(self, sparsity: np.ndarray):
        self.feature_sparsity = sparsity

    def compute_embedding(self, feature_weights: torch.Tensor) -> np.ndarray:
        weights_np = feature_weights.cpu().detach().numpy()
        d_sae = weights_np.shape[0]

        print(f"Computing 2D PCA + {self.property_type} embedding for {d_sae} features...")

        pca = PCA(n_components=2, random_state=self.random_state)
        xy_coords = pca.fit_transform(weights_np)

        if self.property_type == "norm":
            z_coords = np.linalg.norm(weights_np, axis=1, keepdims=True)
        elif self.property_type == "variance":
            z_coords = np.var(weights_np, axis=1, keepdims=True)
        elif self.property_type == "sparsity":
            if self.feature_sparsity is None:
                raise ValueError("Sparsity stats not set. Call set_sparsity_stats() first.")
            z_coords = self.feature_sparsity.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown property_type: {self.property_type}")

        embedding = np.concatenate([xy_coords, z_coords], axis=1)
        print(f"Embedding complete. Shape: {embedding.shape}")
        return embedding

    def get_name(self) -> str:
        return f"PCA-2D+{self.property_type}"


class UMAP2DPlusPropertyEmbedder(FeatureEmbedder):
    def __init__(self, property_type: str = "norm", n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 42):
        """property_type: 'norm', 'variance', or 'sparsity'"""
        self.property_type = property_type
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.feature_sparsity = None

    def set_sparsity_stats(self, sparsity: np.ndarray):
        self.feature_sparsity = sparsity

    def compute_embedding(self, feature_weights: torch.Tensor) -> np.ndarray:
        weights_np = feature_weights.cpu().numpy()
        d_sae = weights_np.shape[0]

        print(f"Computing 2D UMAP + {self.property_type} embedding for {d_sae} features...")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            verbose=True
        )
        xy_coords = reducer.fit_transform(weights_np)

        if self.property_type == "norm":
            z_coords = np.linalg.norm(weights_np, axis=1, keepdims=True)
        elif self.property_type == "variance":
            z_coords = np.var(weights_np, axis=1, keepdims=True)
        elif self.property_type == "sparsity":
            if self.feature_sparsity is None:
                raise ValueError("Sparsity stats not set. Call set_sparsity_stats() first.")
            z_coords = self.feature_sparsity.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown property_type: {self.property_type}")

        embedding = np.concatenate([xy_coords, z_coords], axis=1)
        print(f"Embedding complete. Shape: {embedding.shape}")
        return embedding

    def get_name(self) -> str:
        return f"UMAP-2D+{self.property_type}"


def get_embedder(strategy: str = "umap3d", **kwargs) -> FeatureEmbedder:
    """Factory function. strategy: 'umap3d', 'pca2d+norm', 'pca2d+variance', 'umap2d+norm', etc."""
    strategy = strategy.lower()

    if strategy == "umap3d":
        return UMAP3DEmbedder(**kwargs)
    elif strategy.startswith("pca2d+"):
        property_type = strategy.split("+")[1]
        return PCA2DPlusPropertyEmbedder(property_type=property_type, **kwargs)
    elif strategy.startswith("umap2d+"):
        property_type = strategy.split("+")[1]
        return UMAP2DPlusPropertyEmbedder(property_type=property_type, **kwargs)
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
