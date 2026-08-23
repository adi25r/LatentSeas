package com.latentseas.mod.world;

import com.latentseas.mod.data.LatentSeasSavedData;
import net.minecraft.core.BlockPos;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;

import java.util.ArrayList;
import java.util.List;

/**
 * Marks the results of a probe with a real vanilla beacon beam above each hit blob - visible
 * from far away, which a tiny colored dot never was in the old web game. A single 3x3 iron
 * layer is enough to power a beacon's beam (no need to open its GUI or pick an effect for the
 * light itself to render). Placed directly via level.setBlock, not simulated player
 * placement, so BlockTrackingHandler's break/place listeners never see these - a probe can't
 * accidentally boost a feature.
 */
public final class ProbeBeacon {
    private ProbeBeacon() {}

    // Clearance above the feature's own block before the iron pyramid starts - needs to be
    // generous enough that the beacon reads as "marking that spot from above" rather than
    // sitting right on top of it.
    private static final int CLEARANCE = 5;

    private static final List<BlockPos> placed = new ArrayList<>();

    public static void show(MinecraftServer server, List<Integer> featureIdxs) {
        ServerLevel level = server.getLevel(WorldConstants.FEATURE_WORLD);
        if (level == null) return;
        LatentSeasSavedData data = LatentSeasSavedData.get(server);

        clear(level);

        for (int idx : featureIdxs) {
            BlockPos origin = data.getBlobOrigin(idx);
            if (origin == null) continue;

            BlockPos pyramidLayer = origin.above(CLEARANCE);
            BlockPos beaconPos = pyramidLayer.above();

            for (int dx = -1; dx <= 1; dx++) {
                for (int dz = -1; dz <= 1; dz++) {
                    BlockPos p = pyramidLayer.offset(dx, 0, dz);
                    level.setBlock(p, Blocks.IRON_BLOCK.defaultBlockState(), Block.UPDATE_CLIENTS);
                    placed.add(p.immutable());
                }
            }
            level.setBlock(beaconPos, Blocks.BEACON.defaultBlockState(), Block.UPDATE_CLIENTS);
            placed.add(beaconPos.immutable());
        }
    }

    public static void clear(ServerLevel level) {
        for (BlockPos p : placed) {
            level.setBlock(p, Blocks.AIR.defaultBlockState(), Block.UPDATE_CLIENTS);
        }
        placed.clear();
    }
}
