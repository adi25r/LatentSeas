# LatentSeas

An interactive browser game for exploring the latent space of GPT-2 using Sparse Autoencoder (SAE) features.

## Core Concept

Navigate the "landscape" of GPT-2's latent space by placing "flags" that activate specific SAE features. The model generates outputs based on your flag placements, creating interesting and emergent behaviors.

**The 3D World:**
- **(x, y)** = ground position, from a UMAP embedding of the SAE decoder directions
- **height** = feature *density*, as a kernel density estimate
- Every feature drops a pile of sand; the piles overlap into a continuous surface, so
  clusters of related features become hills you can navigate by
- KDE bandwidth controls peakiness: small is spiky, large is rolling. Both live in
  `src/backend/api.py` (`WORLD_SIZE`, `BANDWIDTH`)

**Gameplay:**
1. You're given a target sentence
2. Probe it - the SAE lights up its feature locations on the map
3. Navigate the terrain and place flags at the lit locations
4. Generate text steered by your placed flags
5. Score on semantic similarity to the target

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Web Demo

1. **Start the backend API:**
```bash
./run_backend.sh
```
The API will start on http://localhost:8000 and load the model (this may take a minute).

2. **In a new terminal, start the frontend:**
```bash
./run_frontend.sh
```

3. **Open your browser to:**
```
http://localhost:3000
```

### Test the Core Model

Test the model functionality directly:
```bash
cd src/model
python test.py
```

## Core Model Usage

```python
from model import LatentExplorer

# Initialize the explorer
explorer = LatentExplorer(model_name="gpt2-small", device="cpu")

# Load SAE features
explorer.load_sae()

# See which features fire when the model reads this text (BOS excluded)
top_features = explorer.print_top_features("the rabid dog", top_k=10)

# Steer generation. Strength is in residual-norm units, not a multiplier:
# ~20-50 is the range these features naturally fire at.
steered = explorer.generate_with_boosted_features(
    prompt="Once upon a time",
    boosted_features={
        3986: 40.0,
        4256: 26.0,
    },
    max_new_tokens=50
)
```

## Features

- Load GPT-2 with SAE features from any layer
- Analyze which features activate for given text
- Steer generation by adding feature directions into the residual stream
- Get detailed statistics about feature activations
- Interactive 3D terrain visualization of the feature landscape
- Web-based demo with real-time feature exploration

## API Endpoints

The backend provides a RESTful API:

- `GET /pointmap` - Get 3D coordinates of all features
- `POST /probe` - Analyze a sentence and get activated features
  ```json
  {"sentence": "Hello world", "threshold": 1.0}
  ```
- `POST /flag` - Place a flag at a feature
  ```json
  {"feature_idx": 3986, "strength": 40.0}
  ```
- `DELETE /flag/{feature_idx}` - Remove a flag
- `GET /flags` - Get all placed flags
- `POST /generate` - Generate text with boosted features
  ```json
  {"prompt": "Once upon a time", "max_tokens": 50, "temperature": 0.7}
  ```
- `POST /score` - Semantic similarity between a generation and a target
  ```json
  {"generated": "a dog barking loudly", "target": "the rabid dog barked", "prompt": "Once upon a time"}
  ```
- `DELETE /flags/clear` - Clear all flags

`POST /generate` also accepts an optional `target`; when present the response carries
`score`, `target`, and `scored_text`.

## How to Play

1. **Probe a sentence** - Enter any sentence to see which features light up on the map
2. **Navigate the terrain** - Use W/A/S/D to walk around, right-click drag to look
3. **Place flags** - Left-click on red (activated) features to place flags
4. **Generate** - Hit generate to see what text the model creates with your placed flags
5. **Experiment** - Try different combinations and see what happens!

## How Steering Works (and how it broke)

The first version of this produced nonsense at every setting. Four separate bugs, all
now fixed and pinned by `src/model/diagnose.py`:

**1. The probe returned the same features for every sentence.** Ranking took the max
activation across all positions, including BOS. The BOS residual stream is an outlier
that makes the SAE fire a fixed set of features at ~420 magnitude, while real content
features fire at ~20-50. So BOS won every ranking, and `"the rabid dog"` and
`"quantum chromodynamics"` both returned features `[23123, 979, 316, 7496, 23111]`.
Fixed by dropping position 0 in `get_feature_activations`.

**2. Boosting multiplied by zero.** Steering did `feature_acts[:, :, idx] *= n`. SAE
activations are sparse — only ~0.17% of the 24,576 features are on at any position. A
feature that is off is exactly `0.0`, and `0 * 30` is still `0`. The only place those
BOS features were nonzero was BOS itself, so the multiplier injected a ~6000-magnitude
spike at the one position that derails generation. Bigger multiplier, worse output.
Steering is now additive: `resid += strength * W_dec[feature]`, skipping BOS.

**3. The residual stream was replaced by its reconstruction.** The hook returned
`sae.decode(sae.encode(x))` on every forward pass. That round trip loses ~24% of the
residual norm (cosine 0.966) and raised CE loss from 2.16 to 2.45 *with no features
selected*. The fix adds a vector rather than substituting a lossy reconstruction, so
zero flags is now exactly the clean model.

**4. The map only held features 0-499.** Probes surface features anywhere in 0-24,575,
so flagged features were almost never on the map. The pointmap now covers all features
and is cached to `src/backend/pointmap_cache.npy` (~20s on first startup).

### Picking a strength

`W_dec` rows are unit norm, so strength is in residual-norm units and is directly
comparable to how hard a feature naturally fires:

- **20-50** — the natural range. Coherent English, clearly on-topic.
- **60-80** — strong steering, grammar starts to fray.
- **100+** — degenerate. At 200 you get `dog dog dog dog dog`.

The probe reports each feature's own activation as `suggested_strength`, which is a good
default: it asks the feature to fire about as hard as it did in your probe sentence.

### It was never the model

Scaling to a bigger model would not have helped, and GPT-3 has neither open weights nor
public SAEs. GPT-2 small steers fine once the pipeline is correct — probing
`"she wept bitterly at the funeral"` and generating from `"Once upon a time"` now yields
*"when the body of a person is for home for the family of loved one"*. If you do want a
larger model later, the realistic option is Gemma-2-2b with GemmaScope SAEs, which
`sae_lens` already supports.

## Scoring

Word error rate is the wrong measure here: these features are semantic directions, not
word triggers, so an on-target run rarely reproduces the target's exact wording. Scoring
is cosine similarity instead (`src/model/scoring.py`).

The embeddings are GPT-2's own token table `W_E` — a word2vec-style matrix that is
already loaded, so nothing extra is downloaded. Mean-pooling it raw does not work: every
pair lands between 0.58 and 0.84 because frequent function words dominate the average,
and "she wept at the funeral" vs "compile the kernel" scored 0.68. Two corrections fix it:

1. subtract the vocabulary-mean embedding, removing the shared common direction
2. drop stopword tokens and unit-normalize the rest before averaging

That widens the gap between paraphrases and unrelated text from 0.09 to 0.16 cosine:

| | score |
|---|---|
| identical text | 100 |
| paraphrase | 35-70 |
| unrelated | 4-16 |

The prompt is stripped before scoring. It is common to every attempt, so leaving it in
inflates all scores and compresses the ranking (measured: 45% narrower spread).

`scoring.py` is modular like `embeddings.py`. `get_scorer("sae", explorer=ex)` swaps in
cosine similarity over SAE feature activations instead, scoring how close you landed in
latent space rather than in word space.

## Terrain and navigation

Height is a KDE over feature positions (`src/model/terrain.py`). Evaluating a gaussian
KDE directly is O(grid x points) — 400M operations for a 128x128 grid over 24k features —
so it bins the points into a 2D histogram and gaussian-blurs it, which is the same thing
for a gaussian kernel and runs in 3ms.

The terrain is computed server-side and sent as a heightmap grid. That matters: the
frontend previously found each vertex's height by scanning every feature for the nearest
one, which is 61M distance checks once the map holds all 24,576 features. It also drew one
mesh per feature; markers are now a single `THREE.Points` draw call.

### Spacing the features out

Scaling the world alone does not make features clickable: UMAP output is clumpy, so a
uniform scale enlarges the clumps too and local crowding is unchanged. Two things run in
sequence (`src/model/terrain.py`):

1. `spread_positions` maps the embedding into a `WORLD_SIZE` square, using percentiles
   rather than min/max so outliers cannot squash everything into the middle
2. `relax_positions` pushes apart any pair closer than `MIN_FEATURE_GAP`, which loosens
   tight clusters while leaving the global layout intact

| | crowded pairs (<0.3 apart) | median gap |
|---|---|---|
| raw scale-up, 120 units | 66.7% | 0.219 |
| + relaxation, 160 units | 3.4% | 0.654 |

Median layout drift from relaxation is 0.2% of the world, so the map still means what the
embedding says. About 1.7% of features stay crowded — those are near-duplicate decoder
directions that cannot be separated without distorting the layout.

Features render as lit 3D icosahedra via a single `InstancedMesh`, so 24k of them are one
draw call and still have real depth and shading. Hovering highlights the feature under the
cursor and names it in the HUD, and probed or flagged features win ties when several are
under the cursor at once.

**Controls:** W/A/S/D move, arrow keys look, Shift sprints, F toggles fly mode, Q/Space
and E go up and down while flying, right-drag is mouse look, left-click a dot flags it.

`node src/frontend/navigation.test.js` checks the movement math by loading `app.js` in a
VM, so it verifies the shipped code rather than a copy. It pins the two bugs that made
navigation feel wrong: the strafe vector was `(cos, 0, -sin)` when the right-handed
`cross(forward, up)` is `(-cos, 0, sin)`, which swapped A and D; and pitch was applied as
negative-is-up, which inverted the up and down arrows.

Run `python src/model/diagnose.py` to re-verify the steering and scoring work, and
`node src/frontend/navigation.test.js` for the navigation math.
