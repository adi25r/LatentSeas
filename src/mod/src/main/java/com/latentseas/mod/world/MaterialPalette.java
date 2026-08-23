package com.latentseas.mod.world;

import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;

public final class MaterialPalette {
    private MaterialPalette() {}

    public static final Block UNKNOWN = Blocks.BIRCH_PLANKS;

    public static final Block GROUND = Blocks.STONE;

    public static final Block BOOST = Blocks.DIAMOND_BLOCK;

    private static final Block[] IDENTITY_PALETTE = {
            Blocks.OAK_PLANKS, Blocks.SPRUCE_PLANKS, Blocks.JUNGLE_PLANKS,
            Blocks.ACACIA_PLANKS, Blocks.DARK_OAK_PLANKS, Blocks.MANGROVE_PLANKS,
            Blocks.CHERRY_PLANKS, Blocks.CRIMSON_PLANKS, Blocks.WARPED_PLANKS,
    };

    public static Block identityFor(int featureIdx) {
        int mixed = Integer.hashCode(featureIdx * -1640531527);
        int idx = Math.floorMod(mixed, IDENTITY_PALETTE.length);
        return IDENTITY_PALETTE[idx];
    }
}
