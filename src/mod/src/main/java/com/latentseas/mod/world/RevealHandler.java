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
 * Walk up to an undiscovered blob and hold sneak (shift) for ~1.2s to reveal what it is -
 * this is the Minecraft translation of the old web game's "hold E to dig", moved off
 * block-breaking entirely per the current design: breaking/placing blocks is now purely
 * the boost/weaken mechanic (BlockTrackingHandler), gated on a blob already being
 * discovered. Sneak was picked over a custom keybinding to avoid the client-side key
 * registration + packet-syncing that would otherwise be needed for a dedicated key.
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

    // Checks the exact block positions around the player's feet via BlobIndex's O(1) lookup,
    // rather than scanning every blob origin for whichever is "nearest within some radius".
    // With feature spacing now as tight as ~2.5 blocks median, a radius search wide enough
    // to reach a feature you're standing next to was also wide enough to catch several
    // other undiscovered neighbours - the "nearest" one flickered between them on tiny
    // movement, so holding sneak by one feature could end up revealing a different one each
    // time. Exact adjacency has no such ambiguity: it only ever finds the feature you're
    // actually standing at.
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
