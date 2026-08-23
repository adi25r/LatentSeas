package com.latentseas.mod.world;

import net.minecraft.core.BlockPos;

import java.util.HashMap;
import java.util.Map;

/**
 * In-memory BlockPos -> feature_idx lookup for every block belonging to a blob, so
 * break/place event handlers can resolve ownership in O(1) rather than scanning ~23k
 * blob origins per interaction. Rebuilt on server start from LatentSeasSavedData - it is
 * derived data, never itself persisted.
 */
public final class BlobIndex {
    private BlobIndex() {}

    private static final Map<Long, Integer> blockToFeature = new HashMap<>();

    public static void clear() {
        blockToFeature.clear();
    }

    public static void registerBlob(int featureIdx, BlockPos origin) {
        int r = WorldConstants.BLOB_RADIUS;
        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -r; dy <= r; dy++) {
                for (int dz = -r; dz <= r; dz++) {
                    if (dx * dx + dy * dy + dz * dz > r * r + 1) continue; // matches the build shape
                    blockToFeature.put(origin.offset(dx, dy, dz).asLong(), featureIdx);
                }
            }
        }
    }

    /** The feature that owns this block, or -1 if it isn't part of any blob. */
    public static int featureAt(BlockPos pos) {
        Integer idx = blockToFeature.get(pos.asLong());
        return idx == null ? -1 : idx;
    }
}
