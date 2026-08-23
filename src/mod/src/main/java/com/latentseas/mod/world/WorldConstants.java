package com.latentseas.mod.world;

import com.latentseas.mod.LatentSeasMod;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.level.Level;

/**
 * The dimension the mod builds into, and the mapping from LatentSeas world-space
 * (as returned by GET /pointmap?profile=minecraft) into Minecraft block coordinates.
 */
public final class WorldConstants {
    private WorldConstants() {}

    public static final ResourceLocation FEATURE_WORLD_ID =
            new ResourceLocation(LatentSeasMod.MODID, "feature_world");
    public static final ResourceKey<Level> FEATURE_WORLD =
            ResourceKey.create(Registries.DIMENSION, FEATURE_WORLD_ID);

    // 1 LatentSeas world unit = 1 block. At the "minecraft" pointmap profile the backend
    // already spaces features ~5-6 units apart (MC_MIN_FEATURE_GAP in api.py), which at
    // this scale reads directly as block spacing - no extra rescaling needed for a v1.
    public static final int Y_BASE = 64;

    public static BlockPos toBlockPos(double worldX, double worldY, double worldZ) {
        return new BlockPos((int) Math.round(worldX), Y_BASE + (int) Math.round(worldY),
                (int) Math.round(worldZ));
    }

    // A single block, not a multi-block blob - cut down after real playtesting showed the
    // 19-block-per-feature version was too much to build/render (23k features x 19 blocks
    // was a meaningful share of a genesis build already under strain from world size alone).
    // r=0 naturally degenerates the surrounding sphere-fill loops in WorldBuilder to just
    // the origin block, so no special-casing was needed elsewhere.
    public static final int BLOB_RADIUS = 0;

    // How many *distinct chunks* the world-builder enters per server tick while draining
    // its queue - not a raw block-edit count. Three real crashes during testing narrowed
    // this down: raw per-tick block budgets (even after moving queue construction off-
    // thread) still let the drain jump between thousands of never-before-generated chunks
    // scattered across the map far faster than Minecraft's async chunk-loading pipeline
    // could keep up, and the main thread stalled waiting on chunk loads (confirmed via a
    // crash stack trace stuck in Level.getChunk). Capping new-chunk entry rate directly,
    // with each chunk's own edits grouped and run together (see WorldBuilder.enqueue),
    // targets the actual bottleneck instead of a proxy for it.
    public static final int CHUNKS_PER_TICK = 8;
}
