package com.latentseas.mod.world;

import net.minecraft.core.BlockPos;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.NbtUtils;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;

/**
 * Gives the player a compass that points at a fixed world position - the same
 * lodestone-linked-compass mechanic vanilla uses (a compass carrying LodestonePos/
 * LodestoneDimension NBT points there instead of spinning), without needing an actual
 * lodestone block anywhere. LodestoneTracked is false since there's no real lodestone for
 * the game to verify still exists.
 */
public final class NavigationCompass {
    private NavigationCompass() {}

    public static void giveTo(ServerPlayer player, ServerLevel level, BlockPos target) {
        ItemStack compass = new ItemStack(Items.COMPASS);
        CompoundTag tag = compass.getOrCreateTag();
        tag.put("LodestonePos", NbtUtils.writeBlockPos(target));
        tag.putString("LodestoneDimension", level.dimension().location().toString());
        tag.putBoolean("LodestoneTracked", false);

        if (!player.getInventory().add(compass)) {
            player.drop(compass, false);
        }
    }
}
