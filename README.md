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

## How Steering Works

`W_dec` rows are unit norm, so strength is in residual-norm units and is directly
comparable to how hard a feature naturally fires:

- **20-50** — the natural range. Coherent English, clearly on-topic.
- **60-80** — strong steering, grammar starts to fray.
- **100+** — degenerate. At 200 you get `dog dog dog dog dog`.

The probe reports each feature's own activation as `suggested_strength`, which is a good
default: it asks the feature to fire about as hard as it did in your probe sentence.

## Scoring

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