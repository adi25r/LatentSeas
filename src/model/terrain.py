"""
A gaussian KDE evaluated directly is O(grid_cells * points), which is 400M operations for
a 128x128 grid over 24k features. Binning into a histogram and gaussian-blurring it is
mathematically the same thing for a gaussian kernel, but runs in milliseconds.
"""
import hashlib
import os

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

    normalized = (xy - lo) / span            
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized - 0.5) * world_size


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

    cells_per_unit = grid_size / world_size
    density = gaussian_filter(density, sigma=bandwidth * cells_per_unit, mode="nearest")

    peak = density.max()
    if peak <= 0:
        return np.zeros_like(density)
    return (density / peak) * max_height


def sample_heightmap(heightmap: np.ndarray, xy_world: np.ndarray,
                     world_size: float = 60.0) -> np.ndarray:
    grid_size = heightmap.shape[0]
    half = world_size / 2.0

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
    """Push apart features that sit on top of each other
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


def sample_surface(heightmap: np.ndarray, xy_world: np.ndarray,
                   world_size: float = 60.0) -> np.ndarray:
    """Height of the surface, matching the renderer's triangulation exactly.

    THREE.PlaneGeometry splits each cell along the anti-diagonal, into triangles
    (a, b, d) and (b, c, d) for a=(r,c) b=(r+1,c) c=(r+1,c+1) d=(r,c+1).
    """
    grid_size = heightmap.shape[0]
    step = world_size / (grid_size - 1)
    half = world_size / 2.0

    gx = np.clip((xy_world[:, 0] + half) / step, 0, grid_size - 1.0 - 1e-9)
    gz = np.clip((xy_world[:, 1] + half) / step, 0, grid_size - 1.0 - 1e-9)

    c0 = gx.astype(int); r0 = gz.astype(int)
    fx = gx - c0; fz = gz - r0

    h_a = heightmap[r0, c0]
    h_b = heightmap[r0 + 1, c0]
    h_c = heightmap[r0 + 1, c0 + 1]
    h_d = heightmap[r0, c0 + 1]

    lower = h_a + fx * (h_d - h_a) + fz * (h_b - h_a)             # fx + fz <= 1
    upper = h_c + (1 - fx) * (h_b - h_c) + (1 - fz) * (h_d - h_c)  # fx + fz >= 1
    return np.where(fx + fz <= 1.0, lower, upper)


def build_terrain(pointmap: np.ndarray, world_size: float, grid_size: int, bandwidth: float,
                  max_height: float, min_gap: float, iterations: int = 30):
    """Embedding -> (ground positions, heightmap, per-feature surface heights)."""
    xy = spread_positions(pointmap[:, :2], world_size=world_size)
    xy = relax_positions(xy, min_dist=min_gap, iterations=iterations, bounds=world_size / 2)
    heights_grid = kde_heightmap(xy, world_size=world_size, grid_size=grid_size,
                                 bandwidth=bandwidth, max_height=max_height)
    return xy, heights_grid, sample_surface(heights_grid, xy, world_size=world_size)


def _terrain_key(pointmap: np.ndarray, params: dict) -> str:
    """Fingerprint of everything the terrain depends on."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(pointmap, dtype=np.float64).tobytes())
    for name in sorted(params):
        h.update(f"{name}={params[name]!r};".encode())
    return h.hexdigest()


def load_or_build_terrain(cache_path: str, pointmap: np.ndarray, **params):
    """Terrain from cache when it is still valid, otherwise rebuilt and cached.
    """
    key = _terrain_key(pointmap, params)

    if os.path.exists(cache_path):
        reason = None
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["key"]) == key:
                return cached["world_xy"], cached["heightmap"], cached["point_heights"], True
            reason = "settings or pointmap changed"
        except Exception as exc:
            reason = f"unreadable ({exc})"
        print(f"  terrain cache stale ({reason}), rebuilding")

    world_xy, heights_grid, point_heights = build_terrain(pointmap, **params)
    try:
        np.savez_compressed(cache_path, world_xy=world_xy, heightmap=heights_grid,
                            point_heights=point_heights, key=key)
    except OSError as exc:
        print(f"  could not write terrain cache: {exc}")
    return world_xy, heights_grid, point_heights, False
