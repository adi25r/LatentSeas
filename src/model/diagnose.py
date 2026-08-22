"""Regression tests for the steering pipeline.

Each test pins down a bug that made the game unplayable. Run from src/model:
    python diagnose.py
"""
import warnings
warnings.filterwarnings("ignore")
import torch
from model import LatentExplorer
from scoring import get_scorer

PROBES = ["the rabid dog", "the ocean waves and sailing ships",
          "banking finance and interest rates", "she wept bitterly at the funeral"]


def test_probe_is_sentence_specific(ex):
    """BOS fires a fixed set of ~400-magnitude features. Including position 0 made
    every sentence return the identical top features, so probing did nothing."""
    tops = {}
    for s in PROBES:
        acts = ex.get_feature_activations(s)
        tops[s] = tuple(torch.topk(acts.max(dim=0).values, 5).indices.tolist())
    distinct = len(set(tops.values()))
    for s, t in tops.items():
        print(f"    {s[:36]:38s} -> {list(t)}")
    assert distinct == len(PROBES), f"only {distinct}/{len(PROBES)} distinct probe results"
    print(f"  PASS: {distinct}/{len(PROBES)} sentences give distinct features")


def test_bos_excluded(ex):
    """BOS activations dwarf content ones; they must not reach the ranking."""
    with_bos = ex.get_feature_activations("the rabid dog", skip_bos=False)
    without = ex.get_feature_activations("the rabid dog", skip_bos=True)
    assert without.shape[0] == with_bos.shape[0] - 1
    print(f"  PASS: BOS dropped. max act {with_bos.max():.0f} (with BOS) "
          f"vs {without.max():.0f} (content only)")


def test_no_flags_is_lossless(ex):
    """Steering must be additive. The old decode(encode(x)) round trip discarded ~24%
    of the residual norm and damaged output even with nothing selected."""
    tokens = ex.model.to_tokens("Once upon a time, there was a small village.")
    clean = ex.model(tokens, return_type="loss").item()
    steer = ex.steering_vector({})
    assert torch.allclose(steer, torch.zeros_like(steer))

    def hook(resid, hook):
        resid[:, 1:, :] += steer
        return resid
    with ex.model.hooks(fwd_hooks=[(ex.hook_point, hook)]):
        steered = ex.model(tokens, return_type="loss").item()
    assert abs(clean - steered) < 1e-4, f"empty steer changed loss {clean} -> {steered}"
    print(f"  PASS: empty steering leaves CE loss at {clean:.4f} (was 2.16 -> 2.45 before)")


def test_steering_changes_output(ex):
    """Strength is in residual-norm units, not a multiplier: a feature that is off at a
    position is exactly 0, so the old `acts *= n` could never turn anything on."""
    prompt = "Once upon a time"
    torch.manual_seed(0)
    baseline = ex.generate_with_boosted_features(prompt, {}, max_new_tokens=22, temperature=0.7)
    outputs = {}
    for s in PROBES:
        feats = {f: a for f, a in
                 zip(*[t.tolist() for t in torch.topk(ex.get_feature_activations(s).max(dim=0).values, 3)][::-1])}
        torch.manual_seed(0)
        outputs[s] = ex.generate_with_boosted_features(prompt, feats, max_new_tokens=22, temperature=0.7)

    print(f"    baseline: {baseline!r}")
    for s, o in outputs.items():
        print(f"    {s[:30]:32s}: {o!r}")
    assert all(o != baseline for o in outputs.values()), "steering did not change output"
    assert len(set(outputs.values())) == len(PROBES), "different probes gave identical output"
    print(f"  PASS: all {len(PROBES)} probes steer to distinct, non-baseline outputs")


def test_scoring_separates_related_from_unrelated(ex):
    """Plain mean-pooled W_E puts everything in 0.58-0.84 because function words dominate.
    Subtracting the vocab mean and dropping stopwords is what creates usable separation."""
    sc = get_scorer("word2vec", model=ex.model)
    assert sc.score("the dog barked", "the dog barked") == 100.0
    assert sc.score("", "the dog barked") == 0.0

    similar = [("the dog barked loudly at the mailman", "a loud dog barking at a postal worker"),
               ("ships sail across the ocean waves", "boats crossing the sea"),
               ("she wept at the funeral", "crying at a burial service")]
    unrelated = [("the dog barked loudly at the mailman", "quantum chromodynamics lagrangian"),
                 ("ships sail across the ocean waves", "interest rates rose at the central bank"),
                 ("she wept at the funeral", "compile the kernel with debug symbols")]

    lo = min(sc.score(a, b) for a, b in similar)
    hi = max(sc.score(a, b) for a, b in unrelated)
    for a, b in similar:
        print(f"    SIMILAR   {sc.score(a,b):5.1f}  {a[:30]:32s} | {b[:30]}")
    for a, b in unrelated:
        print(f"    UNRELATED {sc.score(a,b):5.1f}  {a[:30]:32s} | {b[:30]}")
    assert lo > hi, f"no separation: min(similar)={lo} <= max(unrelated)={hi}"
    print(f"  PASS: margin {lo - hi:.1f} points (similar >= {lo:.1f}, unrelated <= {hi:.1f})")


if __name__ == "__main__":
    ex = LatentExplorer(model_name="gpt2-small", device="cpu")
    ex.load_sae()
    for fn in [test_probe_is_sentence_specific, test_bos_excluded,
               test_no_flags_is_lossless, test_steering_changes_output,
               test_scoring_separates_related_from_unrelated]:
        print(f"\n{fn.__name__}:")
        fn(ex)
    print("\nAll regression tests passed.")
