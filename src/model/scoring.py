import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

STOPWORDS = [
    "the", "a", "an", "of", "and", "or", "to", "in", "at", "on", "for", "with", "that",
    "this", "is", "was", "were", "are", "be", "been", "being", "it", "its", "as", "by",
    "from", "he", "she", "they", "them", "his", "her", "their", "i", "you", "we", "but",
    "not", "no", "so", "if", "then", "than", "there", "here", "what", "which", "who",
    "had", "has", "have", "do", "does", "did", "will", "would", "can", "could", "s", "t",
]


class SimilarityScorer(ABC):
    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """Return a single vector for the text."""

    @abstractmethod
    def get_name(self) -> str:
        pass

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [-1, 1]. Empty input scores 0."""
        if not a.strip() or not b.strip():
            return 0.0
        return F.cosine_similarity(self.embed(a), self.embed(b), dim=0).item()

    def score(self, a: str, b: str) -> float:
        """Similarity as a 0-100 game score."""
        return round(max(0.0, self.similarity(a, b)) * 100, 1)


class Word2VecScorer(SimilarityScorer):
    """Mean-pooled GPT-2 input embeddings, de-biased and stopword-filtered."""

    def __init__(self, model, drop_stopwords: bool = True, normalize_tokens: bool = True):
        self.model = model
        self.drop_stopwords = drop_stopwords
        self.normalize_tokens = normalize_tokens
        self.W_E = model.W_E
        self.vocab_mean = self.W_E.mean(0)

        self.stop_ids = set()
        for word in STOPWORDS:
            for form in (word, " " + word, word.capitalize(), " " + word.capitalize()):
                ids = model.to_tokens(form, prepend_bos=False)[0]
                if len(ids) == 1:
                    self.stop_ids.add(ids[0].item())

    def embed(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            ids = self.model.to_tokens(text)[0][1:].tolist()  # drop BOS
            if self.drop_stopwords:
                ids = [i for i in ids if i not in self.stop_ids] or ids
            if not ids:
                return torch.zeros_like(self.vocab_mean)

            vecs = self.W_E[torch.tensor(ids, device=self.W_E.device)] - self.vocab_mean
            if self.normalize_tokens:
                vecs = F.normalize(vecs, dim=-1)
            return vecs.mean(0)

    def get_name(self) -> str:
        return "gpt2-w2v"


class SAEFeatureScorer(SimilarityScorer):
    """Cosine similarity in the SAE feature space 
    """

    def __init__(self, explorer):
        self.explorer = explorer

    def embed(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            return self.explorer.get_feature_activations(text).max(dim=0).values

    def get_name(self) -> str:
        return "sae-feature"


def get_scorer(strategy: str, model=None, explorer=None, **kwargs) -> SimilarityScorer:
    """Factory. strategy: 'word2vec' or 'sae'."""
    strategy = strategy.lower()
    if strategy in ("word2vec", "w2v", "gpt2-w2v"):
        return Word2VecScorer(model if model is not None else explorer.model, **kwargs)
    if strategy in ("sae", "sae-feature"):
        return SAEFeatureScorer(explorer, **kwargs)
    raise ValueError(f"Unknown scoring strategy: {strategy}")
