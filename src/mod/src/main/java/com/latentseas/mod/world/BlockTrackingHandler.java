package com.latentseas.mod.world;

import com.latentseas.mod.LatentSeasMod;
import com.latentseas.mod.data.LatentSeasSavedData;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.event.entity.player.PlayerInteractEvent;
import net.minecraftforge.event.level.BlockEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;

/**
 * Turns edits on a discovered feature's block into steering state: replacing it with a
 * diamond block (MaterialPalette.BOOST) boosts it; breaking it deactivates the feature.
 * Right-click is a manual on/off toggle at baseline strength. Edits on an undiscovered
 * feature, or anywhere that isn't a feature's block (BlobIndex.featureAt returns -1), do
 * nothing.
 */
public final class BlockTrackingHandler {
    private BlockTrackingHandler() {}

    private static final float DEFAULT_STRENGTH = 40f;
    private static final float BOOST_STRENGTH = 80f;

    @SubscribeEvent
    public static void onBreak(BlockEvent.BreakEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;
        if (!level.dimension().equals(WorldConstants.FEATURE_WORLD)) return;

        int idx = BlobIndex.featureAt(event.getPos());
        if (idx < 0) return;

        LatentSeasSavedData data = LatentSeasSavedData.get(level.getServer());
        if (!data.isDiscovered(idx) || !data.getFlagged().containsKey(idx)) return;

        data.clearFlagged(idx);
        LatentSeasMod.BACKEND.unflag(idx);
        tell(event.getPlayer(), "deactivated feature " + idx);
    }

    @SubscribeEvent
    public static void onPlace(BlockEvent.EntityPlaceEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;
        if (!level.dimension().equals(WorldConstants.FEATURE_WORLD)) return;
        if (!event.getPlacedBlock().is(MaterialPalette.BOOST)) return;

        int idx = BlobIndex.featureAt(event.getPos());
        if (idx < 0) return;

        LatentSeasSavedData data = LatentSeasSavedData.get(level.getServer());
        if (!data.isDiscovered(idx)) return;

        data.setFlagged(idx, BOOST_STRENGTH);
        LatentSeasMod.BACKEND.flag(idx, BOOST_STRENGTH);
        tell(event.getEntity() instanceof Player p ? p : null,
                "feature " + idx + " boosted to " + (int) BOOST_STRENGTH);
    }

    @SubscribeEvent
    public static void onRightClick(PlayerInteractEvent.RightClickBlock event) {
        if (event.getHand() != InteractionHand.MAIN_HAND) return;
        if (!(event.getLevel() instanceof ServerLevel level)) return;
        if (!level.dimension().equals(WorldConstants.FEATURE_WORLD)) return;

        int idx = BlobIndex.featureAt(event.getPos());
        if (idx < 0) return;

        MinecraftServer server = level.getServer();
        LatentSeasSavedData data = LatentSeasSavedData.get(server);
        if (!data.isDiscovered(idx)) return; // nothing to activate until it's been revealed

        event.setCanceled(true);
        Player player = event.getEntity();

        if (data.getFlagged().containsKey(idx)) {
            data.clearFlagged(idx);
            LatentSeasMod.BACKEND.unflag(idx);
            tell(player, "deactivated feature " + idx);
        } else {
            data.setFlagged(idx, DEFAULT_STRENGTH);
            LatentSeasMod.BACKEND.flag(idx, DEFAULT_STRENGTH);
            tell(player, "activated feature " + idx + " at strength " + (int) DEFAULT_STRENGTH);
        }
    }

    private static void tell(Player player, String msg) {
        if (player instanceof ServerPlayer) {
            player.displayClientMessage(Component.literal(msg), true);
        }
    }
}
