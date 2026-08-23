package com.latentseas.mod.world;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;

/**
 * What gives a feature's single block its visual identity (cosmetic, revealed on dig), plus
 * the one block that means anything gameplay-wise: a diamond block boosts a feature, full
 * stop - no tier table, no partial credit for lesser blocks (simplified after playtesting
 * showed the original per-block tier-value system added complexity without adding fun).
 */
public final class MaterialPalette {
    private MaterialPalette() {}

    // Mossy cobblestone was the original pick but reads too close in tone to the stone
    // floor around it - a light plank stands out against gray terrain from a distance,
    // which matters since finding these is the whole point of walking around.
    /** Every undiscovered feature looks like this until POST /dig succeeds on it. */
    public static final Block UNKNOWN = Blocks.BIRCH_PLANKS;

    /** Base terrain surface between features. */
    public static final Block GROUND = Blocks.STONE;

    /** Replacing a discovered feature's block with this boosts it (see BlockTrackingHandler). */
    public static final Block BOOST = Blocks.DIAMOND_BLOCK;

    // A feature's true material - purely cosmetic identity, revealed on discovery. Every
    // vanilla wood species (planks) reads as visually distinct at a glance, which is the
    // point: adjacent features need to be tellable apart on sight. Picked deterministically
    // from feature_idx so the same feature always looks the same across rebuilds, but mixed
    // (not just feature_idx % length) so spatially adjacent indices don't streak into runs
    // of the same wood. Birch is excluded here - it's UNKNOWN above, so a discovered feature
    // never happens to already look identical to an undiscovered one.
    private static final Block[] IDENTITY_PALETTE = {
            Blocks.OAK_PLANKS, Blocks.SPRUCE_PLANKS, Blocks.JUNGLE_PLANKS,
            Blocks.ACACIA_PLANKS, Blocks.DARK_OAK_PLANKS, Blocks.MANGROVE_PLANKS,
            Blocks.CHERRY_PLANKS, Blocks.CRIMSON_PLANKS, Blocks.WARPED_PLANKS,
    };

    public static Block identityFor(int featureIdx) {
        int mixed = Integer.hashCode(featureIdx * -1640531527); // Knuth multiplicative hash
        int idx = Math.floorMod(mixed, IDENTITY_PALETTE.length);
        return IDENTITY_PALETTE[idx];
    }
}
