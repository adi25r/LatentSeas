package com.latentseas.mod.data;

import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.ListTag;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.saveddata.SavedData;

import java.util.*;

/**
 * The mod's durable state, in the world save (region files persist the blocks themselves
 * for free - this is only the bookkeeping needed to make sense of them again on load):
 * where each feature's blob is and what it's really made of, which features have been
 * discovered, and what each is currently flagged at. The Python backend's equivalent state
 * (discovered/placed_flags in api.py) is a plain in-memory dict that resets on restart, so
 * this is the durable copy - on server start the mod replays it back into the backend
 * rather than the other way around.
 */
public class LatentSeasSavedData extends SavedData {
    private static final String NAME = "latentseas";

    private boolean worldBuilt = false;
    private final Map<Integer, BlockPos> blobOrigin = new HashMap<>();
    private final Map<Integer, Block> blobMaterial = new HashMap<>();
    private final Set<Integer> discovered = new HashSet<>();
    private final Map<Integer, Float> flaggedStrength = new HashMap<>();

    public static LatentSeasSavedData get(MinecraftServer server) {
        return server.overworld().getDataStorage().computeIfAbsent(
                LatentSeasSavedData::load, LatentSeasSavedData::new, NAME);
    }

    public boolean isWorldBuilt() {
        return worldBuilt;
    }

    public void setWorldBuilt(boolean built) {
        this.worldBuilt = built;
        setDirty();
    }

    public int getBlobCount() {
        return blobOrigin.size();
    }

    public void putBlob(int featureIdx, BlockPos origin, Block material) {
        blobOrigin.put(featureIdx, origin.immutable());
        blobMaterial.put(featureIdx, material);
        setDirty();
    }

    public BlockPos getBlobOrigin(int featureIdx) {
        return blobOrigin.get(featureIdx);
    }

    public Block getBlobMaterial(int featureIdx) {
        return blobMaterial.get(featureIdx);
    }

    public Map<Integer, BlockPos> allBlobOrigins() {
        return Collections.unmodifiableMap(blobOrigin);
    }

    public boolean isDiscovered(int featureIdx) {
        return discovered.contains(featureIdx);
    }

    public void addDiscovered(int featureIdx) {
        discovered.add(featureIdx);
        setDirty();
    }

    public Set<Integer> getDiscovered() {
        return Collections.unmodifiableSet(discovered);
    }

    public Map<Integer, Float> getFlagged() {
        return Collections.unmodifiableMap(flaggedStrength);
    }

    public void setFlagged(int featureIdx, float strength) {
        flaggedStrength.put(featureIdx, strength);
        setDirty();
    }

    public void clearFlagged(int featureIdx) {
        flaggedStrength.remove(featureIdx);
        setDirty();
    }

    @Override
    public CompoundTag save(CompoundTag tag) {
        tag.putBoolean("worldBuilt", worldBuilt);

        ListTag blobs = new ListTag();
        for (Map.Entry<Integer, BlockPos> e : blobOrigin.entrySet()) {
            CompoundTag b = new CompoundTag();
            b.putInt("idx", e.getKey());
            b.putInt("x", e.getValue().getX());
            b.putInt("y", e.getValue().getY());
            b.putInt("z", e.getValue().getZ());
            ResourceLocation matId = BuiltInRegistries.BLOCK.getKey(blobMaterial.get(e.getKey()));
            b.putString("material", matId.toString());
            blobs.add(b);
        }
        tag.put("blobs", blobs);

        ListTag disc = new ListTag();
        for (int idx : discovered) {
            CompoundTag d = new CompoundTag();
            d.putInt("idx", idx);
            disc.add(d);
        }
        tag.put("discovered", disc);

        ListTag flags = new ListTag();
        for (Map.Entry<Integer, Float> e : flaggedStrength.entrySet()) {
            CompoundTag f = new CompoundTag();
            f.putInt("idx", e.getKey());
            f.putFloat("strength", e.getValue());
            flags.add(f);
        }
        tag.put("flags", flags);

        return tag;
    }

    public static LatentSeasSavedData load(CompoundTag tag) {
        LatentSeasSavedData data = new LatentSeasSavedData();
        data.worldBuilt = tag.getBoolean("worldBuilt");

        ListTag blobs = tag.getList("blobs", CompoundTag.TAG_COMPOUND);
        for (int i = 0; i < blobs.size(); i++) {
            CompoundTag b = blobs.getCompound(i);
            int idx = b.getInt("idx");
            BlockPos pos = new BlockPos(b.getInt("x"), b.getInt("y"), b.getInt("z"));
            Block material = BuiltInRegistries.BLOCK.get(new ResourceLocation(b.getString("material")));
            data.blobOrigin.put(idx, pos);
            data.blobMaterial.put(idx, material == null ? Blocks.STONE : material);
        }

        ListTag disc = tag.getList("discovered", CompoundTag.TAG_COMPOUND);
        for (int i = 0; i < disc.size(); i++) {
            data.discovered.add(disc.getCompound(i).getInt("idx"));
        }

        ListTag flags = tag.getList("flags", CompoundTag.TAG_COMPOUND);
        for (int i = 0; i < flags.size(); i++) {
            CompoundTag f = flags.getCompound(i);
            data.flaggedStrength.put(f.getInt("idx"), f.getFloat("strength"));
        }

        return data;
    }
}
