package com.latentseas.mod.world;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.latentseas.mod.LatentSeasMod;
import com.latentseas.mod.data.LatentSeasSavedData;
import com.mojang.logging.LogUtils;
import net.minecraft.core.BlockPos;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Block;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.slf4j.Logger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

public class WorldBuilder {
    private static final Logger LOGGER = LogUtils.getLogger();

    private static final Map<Long, List<Runnable>> grouped = new LinkedHashMap<>();
    private static final Queue<List<Runnable>> pendingChunks = new ArrayDeque<>();
    private static int totalQueued = 0;
    private static int totalDone = 0;

    private static volatile boolean ready = false;
    private static volatile LatentSeasSavedData buildingData = null;

    public static boolean isReady() {
        return ready;
    }

    public static void begin(MinecraftServer server) {
        LatentSeasSavedData data = LatentSeasSavedData.get(server);

        if (data.isWorldBuilt()) {
            LOGGER.info("LatentSeas world already built ({} blobs) - rebuilding spatial index and resyncing backend state.",
                    data.getBlobCount());
            ready = true;
            BlobIndex.clear();
            data.allBlobOrigins().forEach(BlobIndex::registerBlob);
            resyncBackend(data);
            return;
        }

        LOGGER.info("Building the LatentSeas world for the first time - fetching feature layout...");
        LatentSeasMod.BACKEND.getPointmap("minecraft")
                .thenAccept(json -> onPointmapReady(server, data, json))
                .exceptionally(err -> {
                    LOGGER.error("Could not fetch Minecraft pointmap from backend", err);
                    return null;
                });
    }

    private static void onPointmapReady(MinecraftServer server, LatentSeasSavedData data, JsonObject json) {
        if (json.has("error")) {
            LOGGER.error("Backend returned an error for /pointmap?profile=minecraft: {}", json.get("error"));
            return;
        }

        ServerLevel level = server.getLevel(WorldConstants.FEATURE_WORLD);
        if (level == null) {
            LOGGER.error("Feature dimension '{}' isn't loaded - is data/latentseas/dimension/feature_world.json packaged?",
                    WorldConstants.FEATURE_WORLD_ID);
            return;
        }

        JsonArray points = json.getAsJsonArray("points");
        JsonArray diggable = json.getAsJsonArray("diggable");
        int blobCount = 0;
        for (int i = 0; i < points.size(); i++) {
            if (i >= diggable.size() || !diggable.get(i).getAsBoolean()) continue;
            JsonArray p = points.get(i).getAsJsonArray();
            BlockPos origin = WorldConstants.toBlockPos(
                    p.get(0).getAsDouble(), p.get(1).getAsDouble(), p.get(2).getAsDouble());
            Block material = MaterialPalette.identityFor(i);
            data.putBlob(i, origin, material);
            BlobIndex.registerBlob(i, origin);
            blobCount++;
        }

        Set<Long> occupiedCoarseCells = coarseCellsNearBlobs(data);
        queueBaseTerrain(level, json.getAsJsonArray("heightmap"),
                json.get("grid_size").getAsInt(), json.get("world_size").getAsDouble(),
                occupiedCoarseCells);

        for (BlockPos origin : data.allBlobOrigins().values()) {
            queueBlob(level, origin);
        }

        pendingChunks.addAll(grouped.values());
        int chunkCount = grouped.size();
        grouped.clear();

        buildingData = data;
        int finalBlobCount = blobCount;
        int finalChunkCount = chunkCount;
        server.execute(() -> LOGGER.info(
                "Queued base terrain + {} feature blobs ({} block edits across {} chunks) - "
                        + "entering {} new chunk(s)/tick (~{}s)...",
                finalBlobCount, totalQueued, finalChunkCount, WorldConstants.CHUNKS_PER_TICK,
                finalChunkCount / WorldConstants.CHUNKS_PER_TICK / 20));
    }

    private static final int COARSE_CELL = 32;

    private static long packCell(int cx, int cz) {
        return ((long) cx << 32) ^ (cz & 0xffffffffL);
    }

    private static Set<Long> coarseCellsNearBlobs(LatentSeasSavedData data) {
        Set<Long> occupied = new HashSet<>();
        for (BlockPos origin : data.allBlobOrigins().values()) {
            int cx = Math.floorDiv(origin.getX(), COARSE_CELL);
            int cz = Math.floorDiv(origin.getZ(), COARSE_CELL);
            occupied.add(packCell(cx, cz));
        }
        return occupied;
    }

    private static void queueBaseTerrain(ServerLevel level, JsonArray heightRows, int gridSize,
                                          double worldSize, Set<Long> occupiedCoarseCells) {
        double half = worldSize / 2.0;
        double cell = worldSize / gridSize;
        int span = Math.max(1, (int) Math.round(cell));

        for (int row = 0; row < heightRows.size(); row++) {
            JsonArray rowArr = heightRows.get(row).getAsJsonArray();
            double worldZ = -half + row * cell;
            int z0 = (int) Math.round(worldZ);

            for (int col = 0; col < rowArr.size(); col++) {
                double worldX = -half + col * cell;
                int x0 = (int) Math.round(worldX);

                int coarseCx = Math.floorDiv(x0, COARSE_CELL);
                int coarseCz = Math.floorDiv(z0, COARSE_CELL);
                if (!occupiedCoarseCells.contains(packCell(coarseCx, coarseCz))) continue;

                int y = WorldConstants.Y_BASE + (int) Math.round(rowArr.get(col).getAsDouble());

                for (int dx = 0; dx < span; dx++) {
                    for (int dz = 0; dz < span; dz++) {
                        BlockPos pos = new BlockPos(x0 + dx, y, z0 + dz);
                        enqueue(pos, () -> level.setBlock(pos, MaterialPalette.GROUND.defaultBlockState(),
                                Block.UPDATE_CLIENTS));
                    }
                }
            }
        }
    }

    private static void queueBlob(ServerLevel level, BlockPos origin) {
        int r = WorldConstants.BLOB_RADIUS;
        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -r; dy <= r; dy++) {
                for (int dz = -r; dz <= r; dz++) {
                    if (dx * dx + dy * dy + dz * dz > r * r + 1) continue;
                    BlockPos pos = origin.offset(dx, dy, dz);
                    enqueue(pos, () -> level.setBlock(pos, MaterialPalette.UNKNOWN.defaultBlockState(),
                            Block.UPDATE_CLIENTS));
                }
            }
        }
    }

    private static void enqueue(BlockPos pos, Runnable r) {
        long chunkKey = packCell(pos.getX() >> 4, pos.getZ() >> 4);
        grouped.computeIfAbsent(chunkKey, k -> new ArrayList<>()).add(r);
        totalQueued++;
    }

    private static final int SAVE_INTERVAL_TICKS = 100;
    private static int ticksSinceSave = 0;

    @SubscribeEvent
    public static void onServerTick(TickEvent.ServerTickEvent event) {
        if (event.phase != TickEvent.Phase.END || pendingChunks.isEmpty()) return;

        int chunkBudget = WorldConstants.CHUNKS_PER_TICK;
        while (chunkBudget-- > 0) {
            List<Runnable> chunkEdits = pendingChunks.poll();
            if (chunkEdits == null) {
                LOGGER.info("LatentSeas world build complete ({} block edits placed).", totalDone);
                if (buildingData != null) buildingData.setWorldBuilt(true);
                ready = true;
                event.getServer().saveEverything(true, false, false);
                break;
            }
            for (Runnable r : chunkEdits) {
                r.run();
                totalDone++;
            }
        }

        if (++ticksSinceSave >= SAVE_INTERVAL_TICKS) {
            ticksSinceSave = 0;
            event.getServer().saveEverything(true, false, false);
        }
    }

    private static void resyncBackend(LatentSeasSavedData data) {
        for (int idx : data.getDiscovered()) {
            LatentSeasMod.BACKEND.dig(idx);
        }
        for (Map.Entry<Integer, Float> e : data.getFlagged().entrySet()) {
            LatentSeasMod.BACKEND.flag(e.getKey(), e.getValue());
        }
    }
}
