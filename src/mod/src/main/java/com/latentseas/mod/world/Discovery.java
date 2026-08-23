package com.latentseas.mod.world;

import com.latentseas.mod.LatentSeasMod;
import com.latentseas.mod.data.LatentSeasSavedData;
import com.mojang.logging.LogUtils;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import org.slf4j.Logger;

/** Shared by whatever actually triggers a reveal (RevealHandler's hold-to-uncover). */
final class Discovery {
    private Discovery() {}
    private static final Logger LOGGER = LogUtils.getLogger();

    static void discover(MinecraftServer server, int idx, ServerPlayer player) {
        LatentSeasSavedData data = LatentSeasSavedData.get(server);
        data.addDiscovered(idx);
        LatentSeasMod.BACKEND.dig(idx).thenAccept(json -> server.execute(() -> {
            String label = json.has("label") && !json.get("label").isJsonNull()
                    ? json.get("label").getAsString() : ("feature " + idx);
            revealBlob(server, idx);
            placeSign(server, idx, label);
            if (player != null) {
                player.displayClientMessage(Component.literal("uncovered: " + label), false);
            }
            LOGGER.info("Discovered feature {}: {}", idx, label);
        }));
    }

    private static void placeSign(MinecraftServer server, int idx, String label) {
        ServerLevel level = server.getLevel(WorldConstants.FEATURE_WORLD);
        LatentSeasSavedData data = LatentSeasSavedData.get(server);
        BlockPos origin = data.getBlobOrigin(idx);
        if (level == null || origin == null) return;
        FeatureSign.place(level, origin, label);
    }

    /** Swaps the blob's still-unknown blocks over to its true material - the dug-out block
     *  itself is left broken, so the dig leaves a visible mark rather than looking untouched. */
    private static void revealBlob(MinecraftServer server, int idx) {
        ServerLevel level = server.getLevel(WorldConstants.FEATURE_WORLD);
        LatentSeasSavedData data = LatentSeasSavedData.get(server);
        BlockPos origin = data.getBlobOrigin(idx);
        Block material = data.getBlobMaterial(idx);
        if (level == null || origin == null || material == null) return;

        int r = WorldConstants.BLOB_RADIUS;
        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -r; dy <= r; dy++) {
                for (int dz = -r; dz <= r; dz++) {
                    if (dx * dx + dy * dy + dz * dz > r * r + 1) continue;
                    BlockPos pos = origin.offset(dx, dy, dz);
                    BlockState current = level.getBlockState(pos);
                    if (current.is(MaterialPalette.UNKNOWN)) {
                        level.setBlock(pos, material.defaultBlockState(), Block.UPDATE_CLIENTS);
                    }
                }
            }
        }
    }
}
