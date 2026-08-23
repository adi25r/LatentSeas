from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from model import LatentExplorer
from scoring import get_scorer
from terrain import load_or_build_terrain
from explanations import load_or_fetch, structural_scores
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
descriptions = None
feature_indices = None
placed_flags = {}  # feature_idx -> steering strength
eligible = None
discovered = set()  # feature indices whose identity the player has dug up

class ProbeRequest(BaseModel):
    sentence: str
    threshold: float = 10.0

class FlagRequest(BaseModel):
    feature_idx: int
    strength: float = 40.0


class DigRequest(BaseModel):
    feature_idx: int

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    target: str | None = None      # if set, the response includes a similarity score


class ScoreRequest(BaseModel):
    generated: str
    target: str
    prompt: str | None = None      # stripped before scoring so it cannot inflate the score

POINTMAP_CACHE = os.path.join(os.path.dirname(__file__), "pointmap_semantic.npy")
EXPLANATIONS_CACHE = os.path.join(os.path.dirname(__file__), "explanations_cache.npz")

# Descriptions above this score are about formatting rather than meaning; they fire on
# almost any sentence and are useless (actively harmful) as steering targets.
STRUCTURAL_THRESHOLD = 0.50
TERRAIN_CACHE = os.path.join(os.path.dirname(__file__), "terrain_cache.npz")

# Terrain shape. WORLD_SIZE spreads the features apart for visibility and clicking without
# touching the embedding itself; BANDWIDTH is the KDE sand-pile width, so smaller is peakier.
WORLD_SIZE = 160.0
GRID_SIZE = 128
BANDWIDTH = 3.0
MAX_HEIGHT = 8.0
# Minimum gap between features. Scaling the world alone enlarges the clumps too, so
# crowded pairs are pushed apart directly; the global layout barely moves.
MIN_FEATURE_GAP = 0.7

# Each feature is now a single block (not a multi-block blob), so it no longer needs the
# wide clearance a blob did to avoid swallowing its neighbours - 20x, then 10x, both proved
# too large in playtesting (both a crash risk from the sheer chunk count, and later just
# "too sparse to feel like a world"). 4x keeps real, deliberate walking room between
# features without the world ballooning: area scales with MC_SCALE^2, so dropping from 10x
# to 4x is a ~6x reduction in total terrain footprint too. Built lazily (see
# _get_mc_terrain) since not every session runs the mod, and cached separately so tuning it
# never disturbs the web game's terrain.
MC_SCALE = 4
MC_WORLD_SIZE = WORLD_SIZE * MC_SCALE
MC_GRID_SIZE = 320
MC_BANDWIDTH = BANDWIDTH * MC_SCALE
MC_MAX_HEIGHT = MAX_HEIGHT
MC_MIN_FEATURE_GAP = MIN_FEATURE_GAP * MC_SCALE
MC_TERRAIN_CACHE = os.path.join(os.path.dirname(__file__), "terrain_cache_mc.npz")

world_xy = None
heightmap = None
point_heights = None
mc_world_xy = None
mc_heightmap = None
mc_point_heights = None

@app.on_event("startup")
async def startup_event():
    global explorer, scorer, pointmap, feature_indices, world_xy, heightmap, point_heights
    global descriptions, eligible

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    explorer = LatentExplorer(model_name="gpt2-small", device=device)
    explorer.load_sae()
    scorer = get_scorer("word2vec", model=explorer.model)

    # What each feature actually responds to, in words. Without this the map is 24k
    # anonymous indices; with it every point is a named concept.
    d_sae = explorer.sae.cfg.d_sae
    descriptions, description_vectors, labelled = load_or_fetch(EXPLANATIONS_CACHE, d_sae=d_sae)

    # A feature is worth probing only if we can say what it means and it is not structural.
    structural = structural_scores(description_vectors) if description_vectors.size else np.zeros(d_sae)
    eligible = labelled & (structural <= STRUCTURAL_THRESHOLD)
    print(f"{int(labelled.sum())}/{d_sae} features described, "
          f"{int(eligible.sum())} usable ({int((~eligible).sum())} structural or unlabelled)")

    # The map must cover every feature, since a probe can surface any of them.
    if os.path.exists(POINTMAP_CACHE):
        print(f"Loading cached pointmap from {POINTMAP_CACHE}")
        pointmap = np.load(POINTMAP_CACHE)
    else:
        # Lay the map out over the *descriptions*, not the decoder directions. Embedding
        # raw decoder geometry puts "references to dogs" next to "the end of the document";
        # embedding what the features mean puts it next to cats, bears and pets.
        print(f"Computing semantic pointmap for {d_sae} features (one-time, ~20s)...")
        norms = np.linalg.norm(description_vectors, axis=1, keepdims=True)
        unit = description_vectors / np.maximum(norms, 1e-9)
        pointmap = explorer.get_pointmap(strategy="umap3d", vectors=unit, n_neighbors=15)
        np.save(POINTMAP_CACHE, pointmap)
        print(f"Cached pointmap to {POINTMAP_CACHE}")

    feature_indices = list(range(pointmap.shape[0]))

    # Ground positions come from the embedding; height is feature density, so clusters
    # of related features read as hills you can navigate by. Cached between launches, and
    # the cache key covers every setting below so tuning them rebuilds automatically.
    world_xy, heightmap, point_heights, cached = load_or_build_terrain(
        TERRAIN_CACHE, pointmap,
        world_size=WORLD_SIZE, grid_size=GRID_SIZE, bandwidth=BANDWIDTH,
        max_height=MAX_HEIGHT, min_gap=MIN_FEATURE_GAP, iterations=30,
    )

    print(f"API ready with {len(feature_indices)} features on a "
          f"{WORLD_SIZE:.0f}x{WORLD_SIZE:.0f} terrain (KDE bandwidth {BANDWIDTH}"
          f"{', cached' if cached else ', built'})")

@app.get("/")
async def root():
    return {"message": "LatentSeas API", "status": "running"}

def _get_mc_terrain():
    """Blob-spaced terrain for the Minecraft mod, built on first request and cached to
    disk from then on (see MC_* constants above for why this can't just reuse world_xy)."""
    global mc_world_xy, mc_heightmap, mc_point_heights
    if mc_world_xy is None:
        # 30 iterations (the web game's setting) isn't enough to fully untangle a genuine
        # pileup - some features start out numerically identical (near-duplicate
        # description embeddings) or clipped to the same world-square edge by
        # spread_positions's percentile clamp, and repulsion from ~0 apart to 14 apart
        # takes real iteration count. This build is cached to disk, so paying for more
        # iterations once is cheap relative to leaving 2% of blobs overlapping forever.
        mc_world_xy, mc_heightmap, mc_point_heights, cached = load_or_build_terrain(
            MC_TERRAIN_CACHE, pointmap,
            world_size=MC_WORLD_SIZE, grid_size=MC_GRID_SIZE, bandwidth=MC_BANDWIDTH,
            max_height=MC_MAX_HEIGHT, min_gap=MC_MIN_FEATURE_GAP, iterations=200,
        )
        print(f"Minecraft terrain ready ({'cached' if cached else 'built'})")
    return mc_world_xy, mc_heightmap, mc_point_heights

@app.get("/pointmap")
async def get_pointmap(profile: str = "web"):
    """Feature positions in world space plus the KDE terrain they sit on.

    profile="minecraft" returns the same shape over a far more widely spaced layout, so
    each feature's solid block blob has room to exist without swallowing its neighbours.
    """
    if pointmap is None:
        return {"error": "Pointmap not yet generated"}

    if profile == "minecraft":
        xy, hmap, heights = _get_mc_terrain()
        world_size, max_height, bandwidth, min_gap = (
            MC_WORLD_SIZE, MC_MAX_HEIGHT, MC_BANDWIDTH, MC_MIN_FEATURE_GAP)
    else:
        xy, hmap, heights = world_xy, heightmap, point_heights
        world_size, max_height, bandwidth, min_gap = (
            WORLD_SIZE, MAX_HEIGHT, BANDWIDTH, MIN_FEATURE_GAP)

    # x, ground height, y - already in world space so the client does no layout work
    points = np.column_stack([xy[:, 0], heights, xy[:, 1]])

    return {
        "points": np.round(points, 3).tolist(),
        "count": int(points.shape[0]),
        "heightmap": np.round(hmap, 3).tolist(),
        "grid_size": MC_GRID_SIZE if profile == "minecraft" else GRID_SIZE,
        "world_size": world_size,
        "max_height": max_height,
        "bandwidth": bandwidth,
        "min_gap": min_gap,
        # Only what the player has dug up. Everything else is an unmarked mound, which is
        # the point - the map is a thing to survey, not a labelled index.
        "known": {str(i): descriptions[i] for i in discovered},
        "diggable": eligible.tolist() if eligible is not None else []
    }

@app.post("/probe")
async def probe_sentence(request: ProbeRequest):
    """Analyze a sentence and return activated feature locations"""
    if explorer is None:
        return {"error": "Model not loaded"}

    sentence = request.sentence.strip()
    if not sentence or len(sentence.split()) > 1:
        return {"error": "probe a single word only"}

    # Ranked by raw activation among non-structural features. BOS is excluded inside
    # get_feature_activations; without that every sentence returns the same handful of
    # huge BOS artifacts. There is no "background" text to rank against - eligible already
    # excludes formatting features by their actual description-embedding direction.
    ranked = explorer.probe(sentence, top_k=5, eligible=eligible)

    activated = []
    for idx, act in ranked:
        if act < request.threshold:
            continue
        activated.append({
            "feature_idx": idx,
            # withheld until dug up
            "label": descriptions[idx] if idx in discovered else None,
            "discovered": idx in discovered,
            "activation": round(act, 2),
            # what this feature naturally fires at here - a sane default steering strength
            "suggested_strength": round(act, 2),
            "position": [round(float(world_xy[idx, 0]), 3),
                         round(float(point_heights[idx]), 3),
                         round(float(world_xy[idx, 1]), 3)]
        })

    return {
        "sentence": sentence,
        "activated_features": activated,
        "count": len(activated)
    }

@app.post("/flag")
async def place_flag(request: FlagRequest):
    """Place or update a flag at a feature location"""
    if request.feature_idx not in feature_indices:
        return {"error": "Invalid feature index"}
    if request.feature_idx not in discovered:
        return {"error": "Dig this one up first", "feature_idx": request.feature_idx}

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
                "label": descriptions[idx] if idx in discovered else None,
                "strength": strength,
                "position": [round(float(world_xy[idx, 0]), 3),
                             round(float(point_heights[idx]), 3),
                             round(float(world_xy[idx, 1]), 3)]
            }
            for idx, strength in placed_flags.items()
        ],
        "count": len(placed_flags)
    }

@app.post("/dig")
async def dig(request: DigRequest):
    """Reveal what a feature responds to. The client gates this on standing next to it."""
    idx = request.feature_idx
    if idx not in feature_indices:
        return {"error": "Invalid feature index"}
    if eligible is not None and not eligible[idx]:
        return {"error": "Nothing but formatting noise buried here", "feature_idx": idx}

    first_time = idx not in discovered
    discovered.add(idx)
    return {
        "feature_idx": idx,
        "label": descriptions[idx],
        "newly_discovered": first_time,
        "total_discovered": len(discovered)
    }


@app.get("/discovered")
async def get_discovered():
    return {"features": {str(i): descriptions[i] for i in sorted(discovered)},
            "count": len(discovered)}


@app.delete("/discovered")
async def reset_discovered():
    """Start a fresh run with the map unknown again."""
    discovered.clear()
    placed_flags.clear()
    return {"success": True}


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
