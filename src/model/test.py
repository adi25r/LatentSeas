from model import LatentExplorer
import torch
import matplotlib.pyplot as plt


def test_model_loading():
    """Test that model and SAE load correctly"""
    print("\n" + "="*60)
    print("TEST: Model Loading")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    explorer = LatentExplorer(model_name="gpt2-small", device=device)
    explorer.load_sae()

    assert explorer.sae is not None
    assert explorer.model is not None
    print("✓ Model and SAE loaded successfully")

    return explorer


def test_feature_activations(explorer):
    """Test feature activation extraction"""
    print("\n" + "="*60)
    print("TEST: Feature Activations")
    print("="*60)

    text = "The quick brown fox"
    feature_acts = explorer.get_feature_activations(text)

    assert feature_acts.shape[0] > 0  # Has sequence length
    assert feature_acts.shape[1] == explorer.sae.cfg.d_sae  # Has all features
    print(f"✓ Feature activations shape: {feature_acts.shape}")

    top_features = explorer.print_top_features(text, top_k=5)
    assert len(top_features) == 5
    print("✓ Top features extracted")

    return top_features


def test_pointmap_subset(explorer, n_features=100):
    """Test pointmap generation for a small subset of features"""
    print("\n" + "="*60)
    print(f"TEST: Pointmap Generation (subset of {n_features} features)")
    print("="*60)

    # Get full decoder weights but only use first n_features
    full_weights = explorer.sae.W_dec
    subset_weights = full_weights[:n_features, :]

    print(f"Subset weights shape: {subset_weights.shape}")

    # Test UMAP 3D
    from embeddings import UMAP3DEmbedder
    embedder = UMAP3DEmbedder(n_neighbors=min(15, n_features-1))
    pointmap_umap = embedder.compute_embedding(subset_weights)

    assert pointmap_umap.shape == (n_features, 3)
    print(f"✓ UMAP 3D pointmap shape: {pointmap_umap.shape}")

    # Test PCA 2D + norm
    from embeddings import PCA2DPlusPropertyEmbedder
    embedder_pca = PCA2DPlusPropertyEmbedder(property_type="norm")
    pointmap_pca = embedder_pca.compute_embedding(subset_weights)

    assert pointmap_pca.shape == (n_features, 3)
    print(f"✓ PCA 2D+norm pointmap shape: {pointmap_pca.shape}")

    return pointmap_umap, pointmap_pca


def visualize_pointmap(pointmap, title="Feature Space", feature_indices=None,
                       activated_features=None):
    """Visualize terrain with features as ground markers. (x,y) = ground position, z = height"""
    from scipy.interpolate import griddata
    import numpy as np

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = pointmap[:, 0], pointmap[:, 1], pointmap[:, 2]

    # Create terrain surface by interpolating z values
    grid_resolution = 50
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='cubic', fill_value=z.mean())

    # Plot terrain surface
    surf = ax.plot_surface(xi_grid, yi_grid, zi_grid, alpha=0.3, cmap='terrain',
                           linewidth=0, antialiased=True)

    # Plot features as markers on the terrain
    colors = ['gray'] * len(x)
    sizes = [30] * len(x)
    alphas = [0.4] * len(x)

    # Highlight activated features
    activated_indices = set()
    if activated_features is not None:
        for feat_idx, activation in activated_features:
            if feature_indices is not None and feat_idx in feature_indices:
                local_idx = list(feature_indices).index(feat_idx)
                colors[local_idx] = 'red'
                sizes[local_idx] = 150 + activation * 100
                alphas[local_idx] = 0.9
                activated_indices.add(local_idx)

    # Plot all features at their terrain positions
    for i in range(len(x)):
        ax.scatter([x[i]], [y[i]], [z[i]], c=colors[i], s=sizes[i],
                  alpha=alphas[i], edgecolors='black', linewidth=0.5)

    # Add vertical lines from activated features to show they're on the ground
    if activated_features is not None:
        z_ground = z.min() - (z.max() - z.min()) * 0.1
        for idx in activated_indices:
            ax.plot([x[idx], x[idx]], [y[idx], y[idx]], [z_ground, z[idx]],
                   'r--', alpha=0.5, linewidth=1)

    ax.set_xlabel('X Position (Ground)', fontsize=10)
    ax.set_ylabel('Y Position (Ground)', fontsize=10)
    ax.set_zlabel('Terrain Height', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')

    # Set viewing angle for better terrain view
    ax.view_init(elev=25, azim=45)

    return fig


def test_activation_visualization(explorer, n_features=100):
    """Visualize which features activate for different inputs"""
    print("\n" + "="*60)
    print("TEST: Activation Visualization")
    print("="*60)

    # Get pointmap for subset
    full_weights = explorer.sae.W_dec
    subset_weights = full_weights[:n_features, :]
    feature_indices = list(range(n_features))

    from embeddings import UMAP3DEmbedder
    embedder = UMAP3DEmbedder(n_neighbors=min(15, n_features-1))
    pointmap = embedder.compute_embedding(subset_weights)

    # Test with different texts
    texts = [
        "The quick brown fox",
        "Once upon a time",
        "Hello world"
    ]

    for text in texts:
        print(f"\nAnalyzing: '{text}'")
        feature_acts = explorer.get_feature_activations(text)

        # Get max activation for each feature in our subset
        max_acts = feature_acts.max(dim=0).values[:n_features]

        # Find top activated features in our subset
        activated = []
        for i, act in enumerate(max_acts):
            if act > 0.1:  # Threshold
                activated.append((i, act.item()))

        print(f"  {len(activated)} features activated above threshold")

        # Visualize
        visualize_pointmap(
            pointmap,
            title=f"Feature Space - '{text}'",
            feature_indices=feature_indices,
            activated_features=activated
        )
        plt.savefig(f"/Users/adityarajeev/code/LatentSeas/pointmap_{text.replace(' ', '_')}.png")
        plt.close()

    print(f"✓ Visualizations saved")


def test_boosting_effect(explorer):
    """Test generation with and without boosting"""
    print("\n" + "="*60)
    print("TEST: Boosting Effect")
    print("="*60)

    prompt = "Once upon a time"

    # Get top features
    top_features = explorer.print_top_features(prompt, top_k=3)

    # Baseline generation
    print("\n--- Baseline (no boosting) ---")
    baseline = explorer.generate_with_boosted_features(
        prompt=prompt,
        boosted_features={},
        max_new_tokens=20,
        temperature=0.8
    )
    print(f"Output: {baseline}")

    # Steer each feature at the strength it naturally fired at in the probe
    boosted_features = {feat_idx: act for feat_idx, act in top_features}
    print(f"\n--- Steering features {list(boosted_features.keys())} at natural strength ---")
    boosted = explorer.generate_with_boosted_features(
        prompt=prompt,
        boosted_features=boosted_features,
        max_new_tokens=20,
        temperature=0.8
    )
    print(f"Output: {boosted}")

    print("\n✓ Boosting test complete")


def main():
    explorer = test_model_loading()
    test_feature_activations(explorer)
    pointmap_umap, pointmap_pca = test_pointmap_subset(explorer, n_features=100)

    print("\n" + "="*60)
    print("Pointmap Statistics")
    print("="*60)
    print(f"UMAP 3D - X range: [{pointmap_umap[:, 0].min():.2f}, {pointmap_umap[:, 0].max():.2f}]")
    print(f"UMAP 3D - Y range: [{pointmap_umap[:, 1].min():.2f}, {pointmap_umap[:, 1].max():.2f}]")
    print(f"UMAP 3D - Z range: [{pointmap_umap[:, 2].min():.2f}, {pointmap_umap[:, 2].max():.2f}]")
    print()
    print(f"PCA 2D+norm - X range: [{pointmap_pca[:, 0].min():.2f}, {pointmap_pca[:, 0].max():.2f}]")
    print(f"PCA 2D+norm - Y range: [{pointmap_pca[:, 1].min():.2f}, {pointmap_pca[:, 1].max():.2f}]")
    print(f"PCA 2D+norm - Z range: [{pointmap_pca[:, 2].min():.2f}, {pointmap_pca[:, 2].max():.2f}]")

    test_activation_visualization(explorer, n_features=100)
    test_boosting_effect(explorer)

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
