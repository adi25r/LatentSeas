from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from model import LatentExplorer
from scoring import get_scorer
from terrain import spread_positions, relax_positions, kde_heightmap, sample_heightmap
import torch
import numpy as np

app = FastAPI(title="LatentSeas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
explorer = None
scorer = None
pointmap = None
feature_indices = None
placed_flags = {}  # feature_idx -> steering strength

class ProbeRequest(BaseModel):
    sentence: str
    threshold: float = 1.0

class FlagRequest(BaseModel):
    feature_idx: int
    strength: float = 40.0

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    target: str | None = None      # if set, the response includes a similarity score


class ScoreRequest(BaseModel):
    generated: str
    target: str
    prompt: str | None = None      # stripped before scoring so it cannot inflate the score

POINTMAP_CACHE = os.path.join(os.path.dirname(__file__), "pointmap_cache.npy")

# Terrain shape. WORLD_SIZE spreads the features apart for visibility and clicking without
# touching the embedding itself; BANDWIDTH is the KDE sand-pile width, so smaller is peakier.
WORLD_SIZE = 160.0
GRID_SIZE = 128
BANDWIDTH = 3.0
MAX_HEIGHT = 8.0
# Minimum gap between features. Scaling the world alone enlarges the clumps too, so
# crowded pairs are pushed apart directly; the global layout barely moves.
MIN_FEATURE_GAP = 0.7

world_xy = None
heightmap = None
point_heights = None

@app.on_event("startup")
async def startup_event():
    global explorer, scorer, pointmap, feature_indices, world_xy, heightmap, point_heights

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    explorer = LatentExplorer(model_name="gpt2-small", device=device)
    explorer.load_sae()
    scorer = get_scorer("word2vec", model=explorer.model)

    # The map must cover every feature, since a probe can surface any of them.
    # UMAP over all 24k features takes a few minutes, so cache it to disk.
    if os.path.exists(POINTMAP_CACHE):
        print(f"Loading cached pointmap from {POINTMAP_CACHE}")
        pointmap = np.load(POINTMAP_CACHE)
    else:
        print(f"Computing pointmap for all {explorer.sae.cfg.d_sae} features (one-time, slow)...")
        pointmap = explorer.get_pointmap(strategy="umap3d", n_neighbors=15)
        np.save(POINTMAP_CACHE, pointmap)
        print(f"Cached pointmap to {POINTMAP_CACHE}")

    feature_indices = list(range(pointmap.shape[0]))

    # Ground positions come from the embedding; height is feature density, so clusters
    # of related features read as hills you can navigate by.
    world_xy = spread_positions(pointmap[:, :2], world_size=WORLD_SIZE)
    world_xy = relax_positions(world_xy, min_dist=MIN_FEATURE_GAP, iterations=30,
                               bounds=WORLD_SIZE / 2)
    heightmap = kde_heightmap(world_xy, world_size=WORLD_SIZE, grid_size=GRID_SIZE,
                              bandwidth=BANDWIDTH, max_height=MAX_HEIGHT)
    point_heights = sample_heightmap(heightmap, world_xy, world_size=WORLD_SIZE)

    print(f"API ready with {len(feature_indices)} features on a "
          f"{WORLD_SIZE:.0f}x{WORLD_SIZE:.0f} terrain (KDE bandwidth {BANDWIDTH})")

@app.get("/")
async def root():
    return {"message": "LatentSeas API", "status": "running"}

@app.get("/pointmap")
async def get_pointmap():
    """Feature positions in world space plus the KDE terrain they sit on"""
    if pointmap is None:
        return {"error": "Pointmap not yet generated"}

    # x, ground height, y - already in world space so the client does no layout work
    points = np.column_stack([world_xy[:, 0], point_heights, world_xy[:, 1]])

    return {
        "points": np.round(points, 3).tolist(),
        "count": int(points.shape[0]),
        "heightmap": np.round(heightmap, 3).tolist(),
        "grid_size": GRID_SIZE,
        "world_size": WORLD_SIZE,
        "max_height": MAX_HEIGHT,
        "bandwidth": BANDWIDTH,
        "min_gap": MIN_FEATURE_GAP
    }

@app.post("/probe")
async def probe_sentence(request: ProbeRequest):
    """Analyze a sentence and return activated feature locations"""
    if explorer is None:
        return {"error": "Model not loaded"}

    # BOS is excluded inside get_feature_activations - without that every sentence
    # returns the same handful of huge BOS-artifact features.
    feature_acts = explorer.get_feature_activations(request.sentence)
    max_acts = feature_acts.max(dim=0).values

    # Rank instead of scanning all 24k in python; a probe is sparse anyway.
    top_values, top_indices = torch.topk(max_acts, k=min(64, max_acts.shape[0]))

    activated = []
    for act, idx in zip(top_values.tolist(), top_indices.tolist()):
        if act <= request.threshold:
            break
        activated.append({
            "feature_idx": idx,
            "activation": act,
            # what this feature naturally fires at here - a sane default steering strength
            "suggested_strength": round(act, 2),
            "position": [round(float(world_xy[idx, 0]), 3),
                         round(float(point_heights[idx]), 3),
                         round(float(world_xy[idx, 1]), 3)]
        })

    return {
        "sentence": request.sentence,
        "activated_features": activated,
        "count": len(activated)
    }

@app.post("/flag")
async def place_flag(request: FlagRequest):
    """Place or update a flag at a feature location"""
    if request.feature_idx not in feature_indices:
        return {"error": "Invalid feature index"}

    placed_flags[request.feature_idx] = request.strength

    return {
        "success": True,
        "feature_idx": request.feature_idx,
        "strength": request.strength,
        "total_flags": len(placed_flags)
    }

@app.delete("/flag/{feature_idx}")
async def remove_flag(feature_idx: int):
    """Remove a flag"""
    if feature_idx in placed_flags:
        del placed_flags[feature_idx]
        return {"success": True, "removed": feature_idx}
    return {"success": False, "error": "Flag not found"}

@app.get("/flags")
async def get_flags():
    """Get all placed flags"""
    return {
        "flags": [
            {
                "feature_idx": idx,
                "strength": strength,
                "position": [round(float(world_xy[idx, 0]), 3),
                             round(float(point_heights[idx]), 3),
                             round(float(world_xy[idx, 1]), 3)]
            }
            for idx, strength in placed_flags.items()
        ],
        "count": len(placed_flags)
    }

def strip_prompt(generated: str, prompt: str | None) -> str:
    """Score only what the model added. The shared prompt is common to every attempt,
    so leaving it in inflates all scores and compresses the ranking."""
    if prompt and generated.startswith(prompt):
        return generated[len(prompt):].lstrip(" ,.\n")
    return generated


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text based on placed flags, optionally scored against a target"""
    if explorer is None:
        return {"error": "Model not loaded"}

    if len(placed_flags) == 0:
        return {"error": "No flags placed"}

    generated = explorer.generate_with_boosted_features(
        prompt=request.prompt,
        boosted_features=placed_flags,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )

    response = {
        "prompt": request.prompt,
        "generated": generated,
        "flags_used": len(placed_flags)
    }

    if request.target:
        continuation = strip_prompt(generated, request.prompt)
        response["score"] = scorer.score(continuation, request.target)
        response["target"] = request.target
        response["scored_text"] = continuation

    return response


@app.post("/score")
async def score_text(request: ScoreRequest):
    """Semantic similarity between a generation and the target, as a 0-100 score"""
    if scorer is None:
        return {"error": "Model not loaded"}

    continuation = strip_prompt(request.generated, request.prompt)
    return {
        "score": scorer.score(continuation, request.target),
        "similarity": round(scorer.similarity(continuation, request.target), 4),
        "scorer": scorer.get_name(),
        "scored_text": continuation
    }

@app.delete("/flags/clear")
async def clear_flags():
    """Clear all placed flags"""
    placed_flags.clear()
    return {"success": True, "message": "All flags cleared"}
