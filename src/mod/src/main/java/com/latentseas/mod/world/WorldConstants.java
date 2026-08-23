package com.latentseas.mod.world;

import com.latentseas.mod.LatentSeasMod;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.level.Level;

public final class WorldConstants {
    private WorldConstants() {}

    public static final ResourceLocation FEATURE_WORLD_ID =
            new ResourceLocation(LatentSeasMod.MODID, "feature_world");
    public static final ResourceKey<Level> FEATURE_WORLD =
            ResourceKey.create(Registries.DIMENSION, FEATURE_WORLD_ID);

    public static final int Y_BASE = 64;

    public static BlockPos toBlockPos(double worldX, double worldY, double worldZ) {
        return new BlockPos((int) Math.round(worldX), Y_BASE + (int) Math.round(worldY),
                (int) Math.round(worldZ));
    }

    public static final int BLOB_RADIUS = 0;

    public static final int CHUNKS_PER_TICK = 8;
}
