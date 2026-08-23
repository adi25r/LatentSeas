import gzip
import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MODEL_PREFERENCE = ["gpt-4o-mini", "gpt-3.5-turbo"]


def _batch_keys(model_id: str, sae_id: str) -> list[str]:
    prefix = f"v1/{model_id}/{sae_id}/explanations/"
    url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        root = ET.fromstring(response.read())

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [k.text for k in root.findall(".//s3:Contents/s3:Key", ns)]
    return sorted(k for k in keys if re.fullmatch(rf"{prefix}batch-\d+\.jsonl\.gz", k or ""))


def fetch_explanations(model_id: str = "gpt2-small", sae_id: str = "6-res-jb",
                       d_sae: int = 24576, verbose: bool = True):
    """Download and collapse to one explanation per feature.

    Returns (descriptions, embeddings, have) where descriptions is a list of length d_sae
    ("" where Neuronpedia has none), embeddings is [d_sae, dim], and have is a bool mask.
    """
    keys = _batch_keys(model_id, sae_id)
    if not keys:
        raise RuntimeError(f"no explanation batches found for {model_id}/{sae_id}")

    descriptions = [""] * d_sae
    chosen_rank = [len(MODEL_PREFERENCE) + 1] * d_sae
    vectors: dict[int, np.ndarray] = {}

    for n, key in enumerate(keys, 1):
        if verbose:
            print(f"  explanations batch {n}/{len(keys)}", end="\r", flush=True)
        with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as response:
            payload = gzip.decompress(response.read())

        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                idx = int(row["index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if not 0 <= idx < d_sae:
                continue

            name = row.get("explanationModelName")
            rank = MODEL_PREFERENCE.index(name) if name in MODEL_PREFERENCE else len(MODEL_PREFERENCE)
            if rank >= chosen_rank[idx]:
                continue

            text = (row.get("description") or "").strip()
            if not text:
                continue

            chosen_rank[idx] = rank
            descriptions[idx] = text
            raw = row.get("embedding")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = None
            if raw:
                vectors[idx] = np.asarray(raw, dtype=np.float32)

    if verbose:
        print(" " * 40, end="\r")

    dim = len(next(iter(vectors.values()))) if vectors else 0
    embeddings = np.zeros((d_sae, dim), dtype=np.float32)
    for idx, vec in vectors.items():
        if len(vec) == dim:
            embeddings[idx] = vec

    have = np.array([bool(d) for d in descriptions])
    return descriptions, embeddings, have


def load_or_fetch(cache_path: str, model_id: str = "gpt2-small", sae_id: str = "6-res-jb",
                  d_sae: int = 24576):
    """Cached wrapper around fetch_explanations. Returns (descriptions, embeddings, have)."""
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if int(cached["d_sae"]) == d_sae:
                return list(cached["descriptions"]), cached["embeddings"], cached["have"]
            print("  explanation cache is for a different SAE size, refetching")
        except Exception as exc:
            print(f"  explanation cache unreadable ({exc}), refetching")

    print(f"Fetching Neuronpedia explanations for {model_id}/{sae_id} (one-time, ~50MB)...")
    try:
        descriptions, embeddings, have = fetch_explanations(model_id, sae_id, d_sae)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        print(f"  could not reach Neuronpedia ({exc}); features stay unlabelled")
        return [""] * d_sae, np.zeros((d_sae, 0), dtype=np.float32), np.zeros(d_sae, bool)

    try:
        np.savez_compressed(cache_path, descriptions=np.array(descriptions, dtype=object).astype("U"),
                            embeddings=embeddings, have=have, d_sae=d_sae)
    except OSError as exc:
        print(f"  could not write explanation cache: {exc}")

    print(f"  {int(have.sum())}/{d_sae} features labelled")
    return descriptions, embeddings, have

STRUCTURAL_SEEDS = [4256, 9886, 20314, 7811, 3252]

def structural_scores(embeddings: np.ndarray, seeds=None) -> np.ndarray:
    """How much each feature's description is about formatting rather than content.

    Some SAE features detect delimiters, whitespace, the definite article, end-of-text
    markers. They fire strongly on ordinary sentences and rank near the top of every probe,
    but steering toward "text marker symbols" only produces noise. Scoring each description
    against the centroid of a few known structural ones separates them cleanly: they land
    around 0.72-0.77, while concept features like "references to dogs" sit near 0.36.
    """
    seeds = STRUCTURAL_SEEDS if seeds is None else seeds
    if embeddings.size == 0:
        return np.zeros(len(embeddings))

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norm, 1e-9)
    centroid = unit[seeds].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)
    return unit @ centroid
