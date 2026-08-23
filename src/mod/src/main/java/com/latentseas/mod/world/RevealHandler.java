package com.latentseas.mod.world;

import com.latentseas.mod.data.LatentSeasSavedData;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * Hold sneak (shift) next to an undiscovered feature for ~1.2s to reveal it.
 */
public final class RevealHandler {
    private RevealHandler() {}

    private static final float HOLD_SECONDS = 1.2f;
    private static final float TICK_STEP = 1f / 20f;

    private static final Map<UUID, Integer> target = new HashMap<>();
    private static final Map<UUID, Float> progress = new HashMap<>();

    @SubscribeEvent
    public static void onPlayerTick(TickEvent.PlayerTickEvent event) {
        if (event.phase != TickEvent.Phase.END) return;
        if (!(event.player instanceof ServerPlayer player)) return;
        if (!player.level().dimension().equals(WorldConstants.FEATURE_WORLD)) return;

        UUID id = player.getUUID();
        int near = nearestUndiscoveredBlob(player);

        if (!player.isShiftKeyDown() || near < 0) {
            progress.remove(id);
            target.remove(id);
            return;
        }

        if (!Objects.equals(target.get(id), near)) {
            target.put(id, near);
            progress.put(id, 0f);
        }

        float p = progress.getOrDefault(id, 0f) + TICK_STEP;
        if (p >= HOLD_SECONDS) {
            progress.remove(id);
            target.remove(id);
            Discovery.discover(player.getServer(), near, player);
        } else {
            progress.put(id, p);
            // actionbar every few ticks rather than every tick - still reads as continuous
            if (((int) (p * 20)) % 4 == 0) {
                int pct = Math.round(p / HOLD_SECONDS * 100);
                player.displayClientMessage(Component.literal("uncovering... " + pct + "%"), true);
            }
        }
    }

    private static int nearestUndiscoveredBlob(ServerPlayer player) {
        LatentSeasSavedData data = LatentSeasSavedData.get(player.getServer());
        BlockPos feet = player.blockPosition();

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int idx = BlobIndex.featureAt(feet.offset(dx, dy, dz));
                    if (idx >= 0 && !data.isDiscovered(idx)) return idx;
                }
            }
        }
        return -1;
    }
}
