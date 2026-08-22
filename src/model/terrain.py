"""Turn a 2D feature embedding into a continuous, navigable terrain.

Height is a kernel density estimate over the feature positions: every feature drops a pile
of sand, the piles overlap into a smooth surface, and dense clusters of related features
become hills. Bandwidth controls how peaky the result is - small bandwidth gives sharp
spikes per cluster, large bandwidth gives broad rolling hills.

A gaussian KDE evaluated directly is O(grid_cells * points), which is 400M operations for
a 128x128 grid over 24k features. Binning into a histogram and gaussian-blurring it is
mathematically the same thing for a gaussian kernel, but runs in milliseconds.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def spread_positions(xy: np.ndarray, world_size: float = 60.0,
                     percentile: float = 1.0) -> np.ndarray:
    """Map embedding coordinates into a world-space square.

    UMAP output has outliers that would squash everything else into the middle, so the
    range is taken from percentiles rather than min/max, then clipped.
    """
    lo = np.percentile(xy, percentile, axis=0)
    hi = np.percentile(xy, 100 - percentile, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    normalized = (xy - lo) / span            # roughly [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized - 0.5) * world_size   # centered on the origin


def kde_heightmap(xy_world: np.ndarray, world_size: float = 60.0, grid_size: int = 128,
                  bandwidth: float = 2.0, max_height: float = 8.0) -> np.ndarray:
    """Grid of terrain heights from feature density.

    Args:
        xy_world: [n, 2] positions already in world space, centered on the origin
        world_size: width of the terrain square
        grid_size: heightmap resolution per axis
        bandwidth: sand pile width in world units - larger is smoother and less peaky
        max_height: tallest peak after normalization

    Returns:
        [grid_size, grid_size] array of heights, indexed [row=y, col=x]
    """
    half = world_size / 2.0
    edges = np.linspace(-half, half, grid_size + 1)

    density, _, _ = np.histogram2d(
        xy_world[:, 1], xy_world[:, 0], bins=[edges, edges]  # row=y, col=x
    )

    # sigma is in grid cells, so convert the bandwidth from world units
    cells_per_unit = grid_size / world_size
    density = gaussian_filter(density, sigma=bandwidth * cells_per_unit, mode="nearest")

    peak = density.max()
    if peak <= 0:
        return np.zeros_like(density)
    return (density / peak) * max_height


def sample_heightmap(heightmap: np.ndarray, xy_world: np.ndarray,
                     world_size: float = 60.0) -> np.ndarray:
    """Bilinearly sample terrain heights at arbitrary positions, so markers sit on the
    surface rather than floating above or sinking into it."""
    grid_size = heightmap.shape[0]
    half = world_size / 2.0

    # to continuous grid coordinates, offset by half a cell to hit cell centers
    g = (xy_world + half) / world_size * grid_size - 0.5
    g = np.clip(g, 0, grid_size - 1)

    x0 = np.floor(g[:, 0]).astype(int); x1 = np.minimum(x0 + 1, grid_size - 1)
    y0 = np.floor(g[:, 1]).astype(int); y1 = np.minimum(y0 + 1, grid_size - 1)
    fx = g[:, 0] - x0
    fy = g[:, 1] - y0

    top = heightmap[y0, x0] * (1 - fx) + heightmap[y0, x1] * fx
    bot = heightmap[y1, x0] * (1 - fx) + heightmap[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def relax_positions(xy: np.ndarray, min_dist: float, iterations: int = 14,
                    strength: float = 0.6, bounds: float | None = None) -> np.ndarray:
    """Push apart features that sit on top of each other, so they can be seen and clicked.

    Scaling the world up alone does not help: UMAP output is clumpy, so a uniform scale
    enlarges the clumps too and the local crowding is unchanged. This only acts on pairs
    closer than min_dist, so tight clusters loosen while the global layout is preserved.
    """
    pts = xy.astype(np.float64).copy()

    # exact duplicates have no direction to separate along, so nudge them apart first
    rng = np.random.default_rng(0)
    pts += rng.normal(0, min_dist * 1e-3, pts.shape)

    for _ in range(iterations):
        tree = cKDTree(pts)
        pairs = tree.query_pairs(min_dist, output_type="ndarray")
        if len(pairs) == 0:
            break

        i, j = pairs[:, 0], pairs[:, 1]
        delta = pts[j] - pts[i]
        dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        direction = delta / dist
        push = (min_dist - dist) * 0.5 * strength

        disp = np.zeros_like(pts)
        np.add.at(disp, i, -direction * push)
        np.add.at(disp, j, direction * push)
        pts += disp

        if bounds is not None:
            np.clip(pts, -bounds, bounds, out=pts)

    return pts
