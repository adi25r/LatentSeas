"""API-level tests for the discovery loop. Run from src/backend:  python test_api.py"""
import warnings
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient
import api


def test_probe_withholds_identities(c):
    """Probing marks where to dig. Handing over the labels too would leave nothing to do
    with the map, which is the whole reason the terrain exists."""
    c.delete("/discovered")
    assert len(c.get("/pointmap").json()["known"]) == 0

    sites = c.post("/probe", json={"sentence": "the rabid dog barked"}).json()["activated_features"]
    assert sites, "probe returned nothing"
    for s in sites[:5]:
        assert s["label"] is None and not s["discovered"], "probe leaked a label"
        assert len(s["position"]) == 3
    print(f"  PASS: {len(sites)} sites marked, none named")


def test_flag_requires_digging(c):
    """You cannot steer with a feature you have not identified."""
    c.delete("/discovered")
    idx = c.post("/probe", json={"sentence": "the rabid dog barked"}
                 ).json()["activated_features"][0]["feature_idx"]

    refused = c.post("/flag", json={"feature_idx": idx, "strength": 40}).json()
    assert "error" in refused, "flagging an undug feature should be refused"
    assert c.get("/flags").json()["count"] == 0

    c.post("/dig", json={"feature_idx": idx})
    assert c.post("/flag", json={"feature_idx": idx, "strength": 40}).json().get("success")
    print(f"  PASS: flag refused before digging ({refused['error']!r}), accepted after")


def test_digging_reveals_and_persists(c):
    c.delete("/discovered")
    idx = c.post("/probe", json={"sentence": "the rabid dog barked"}
                 ).json()["activated_features"][0]["feature_idx"]

    first = c.post("/dig", json={"feature_idx": idx}).json()
    assert first["newly_discovered"] and first["label"]
    assert not c.post("/dig", json={"feature_idx": idx}).json()["newly_discovered"]

    # the reveal sticks, on the map and in later probes
    assert str(idx) in c.get("/pointmap").json()["known"]
    again = c.post("/probe", json={"sentence": "the rabid dog barked"}).json()
    row = next(f for f in again["activated_features"] if f["feature_idx"] == idx)
    assert row["discovered"] and row["label"] == first["label"]
    print(f"  PASS: dug up {first['label']!r}, and it stays known")


def test_structural_sites_are_barren(c):
    """Formatting features are filtered from probes, so digging one is a dead end
    rather than a silent success that pollutes the player's flags."""
    out = c.post("/dig", json={"feature_idx": 9886}).json()
    assert "error" in out, "structural feature should not be diggable"
    print(f"  PASS: {out['error']!r}")


def test_full_round(c):
    c.delete("/discovered")
    target = "the rabid dog barked"
    for s in c.post("/probe", json={"sentence": target}).json()["activated_features"][:3]:
        c.post("/dig", json={"feature_idx": s["feature_idx"]})
        c.post("/flag", json={"feature_idx": s["feature_idx"],
                              "strength": s["suggested_strength"]})

    g = c.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 28,
                                  "temperature": 0.7, "target": target}).json()
    print(f"    {g['generated'][:78]!r}")
    assert g["flags_used"] == 3 and "score" in g
    print(f"  PASS: probe -> dig -> flag -> generate, scored {g['score']}")


if __name__ == "__main__":
    with TestClient(api.app) as client:
        for fn in [test_probe_withholds_identities, test_flag_requires_digging,
                   test_digging_reveals_and_persists, test_structural_sites_are_barren,
                   test_full_round]:
            print(f"\n{fn.__name__}:")
            fn(client)
    print("\nAll API tests passed.")
