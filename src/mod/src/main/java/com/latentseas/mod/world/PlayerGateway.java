package com.latentseas.mod.world;

import com.mojang.logging.LogUtils;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.player.PlayerEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.slf4j.Logger;

import java.util.HashSet;
import java.util.Set;

/**
 * Drops a player into the feature dimension on login and on respawn. Waits for
 * WorldBuilder.isReady() first, since a queued-but-undrained world has nothing under the
 * spawn point yet.
 */
public final class PlayerGateway {
    private PlayerGateway() {}
    private static final Logger LOGGER = LogUtils.getLogger();
    private static final Set<ServerPlayer> waiting = new HashSet<>();

    @SubscribeEvent
    public static void onLogin(PlayerEvent.PlayerLoggedInEvent event) {
        if (event.getEntity() instanceof ServerPlayer player) enter(player);
    }

    @SubscribeEvent
    public static void onRespawn(PlayerEvent.PlayerRespawnEvent event) {
        if (event.getEntity() instanceof ServerPlayer player) enter(player);
    }

    private static void enter(ServerPlayer player) {
        if (player.level().dimension().equals(WorldConstants.FEATURE_WORLD)) return;

        if (!WorldBuilder.isReady()) {
            player.displayClientMessage(
                    Component.literal("LatentSeas world is still generating - hold tight..."), false);
            waiting.add(player);
            return;
        }
        teleport(player);
    }

    private static void teleport(ServerPlayer player) {
        ServerLevel level = player.getServer() == null ? null
                : player.getServer().getLevel(WorldConstants.FEATURE_WORLD);
        if (level == null) return;

        int detected = level.getHeight(Heightmap.Types.MOTION_BLOCKING, 0, 0);
        int y = (detected >= WorldConstants.Y_BASE ? detected : WorldConstants.Y_BASE + 8) + 1;

        player.teleportTo(level, 0.5, y, 0.5, 0f, 0f);
        LOGGER.info("Teleported {} into the feature world at (0, {}, 0)", player.getGameProfile().getName(), y);
    }

    @SubscribeEvent
    public static void onServerTick(TickEvent.ServerTickEvent event) {
        if (event.phase != TickEvent.Phase.END || waiting.isEmpty() || !WorldBuilder.isReady()) return;
        for (ServerPlayer player : waiting) teleport(player);
        waiting.clear();
    }
}
