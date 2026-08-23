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

/**
 * Builds the feature world exactly once per save (base terrain from the heightmap grid,
 * then a small blob per eligible feature), queuing every block edit and draining it a few
 * thousand at a time on the server tick so a ~800k-block genesis build never turns into one
 * multi-second tick and trips the watchdog. On every later boot it skips straight to
 * replaying the mod's saved discovered/flag state into the backend, since the backend's
 * own copy is memory-only and resets on restart (see LatentSeasSavedData's class doc).
 */
public class WorldBuilder {
    private static final Logger LOGGER = LogUtils.getLogger();

    // Grouped by chunk (packed chunk x/z -> that chunk's edits) while being built, so every
    // block belonging to the same chunk runs together - see WorldConstants.CHUNKS_PER_TICK
    // for why this grouping exists. LinkedHashMap keeps first-touched order stable, though
    // only per-chunk contiguity actually matters here.
    private static final Map<Long, List<Runnable>> grouped = new LinkedHashMap<>();
    private static final Queue<List<Runnable>> pendingChunks = new ArrayDeque<>();
    private static int totalQueued = 0;
    private static int totalDone = 0;

    // Distinct from LatentSeasSavedData.isWorldBuilt(): that flag means "queued, don't
    // rebuild" and is set as soon as the ~800k edits are queued, well before the tick
    // handler finishes draining them. PlayerGateway needs to know when it's actually safe
    // to drop someone in - teleporting onto a column with nothing placed yet is a fall into
    // the void (this happened during testing: player spawned at Y 84 before any terrain
    // existed and died falling out of the world before the build caught up).
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
            ready = true; // blocks are already on disk from a prior completed build
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

    // Runs on the HTTP client's worker thread, NOT the server thread. Parsing the JSON and
    // constructing ~800k queued Runnables (measured: this alone was enough to trip "Can't
    // keep up" even with a modest per-tick drain budget, since it was all happening inside
    // one server.execute callback) happens here instead, off any single tick. Nothing here
    // touches actual level/chunk state - the Runnables only call level.setBlock when the
    // tick handler later drains them on the main thread, so building the queue itself is
    // safe off-thread.
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

        // Pass 1: record every blob's position/material (data + spatial index) WITHOUT
        // queueing its block placement yet. Positions need to be known before deciding which
        // terrain cells are worth filling (see queueBaseTerrain below) - but queueing must
        // wait until after terrain is queued too. A real bug found in playtesting: a
        // feature's block and its local terrain cell often land on the exact same position
        // (both derived from the same heightmap), and whichever was queued *last* wins that
        // position when the tick handler runs each chunk's edits in order. With blobs queued
        // first, terrain silently overwrote features with plain stone - discoverable (the
        // spatial index still pointed at the right spot) but invisible, since nothing marked
        // it. Terrain is now queued first and blobs second, so a feature's block always wins.
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

        // A real crash during testing traced to vanilla's chunk-unload/save machinery, not
        // our own code: a full-coverage floor across the whole (now much larger) world
        // touches on the order of 40,000 chunks in under a minute, far faster than
        // Minecraft's own chunk lifecycle is built to keep up with, and the save-storm from
        // that pile-up is what tripped the watchdog. Blobs cluster (this is a UMAP layout,
        // not a uniform grid) rather than covering the world evenly, so most of that area
        // has no blob anywhere near it - skip terrain there entirely instead of just
        // generating it slower.
        Set<Long> occupiedCoarseCells = coarseCellsNearBlobs(data);
        queueBaseTerrain(level, json.getAsJsonArray("heightmap"),
                json.get("grid_size").getAsInt(), json.get("world_size").getAsDouble(),
                occupiedCoarseCells);

        // Pass 2: now queue the actual blob block placements, after terrain, so they win
        // any position collision.
        for (BlockPos origin : data.allBlobOrigins().values()) {
            queueBlob(level, origin);
        }

        pendingChunks.addAll(grouped.values());
        int chunkCount = grouped.size();
        grouped.clear();

        // NOT marked worldBuilt yet - that has to wait until the queue actually drains
        // (see onServerTick). Setting it here, when only the *queue* exists and no block
        // has been placed, is exactly what caused a real problem during testing: the
        // server was stopped mid-build, the save was left thinking it was done, and a
        // restart would have skipped straight to resync instead of finishing the build -
        // permanently freezing the world half-built.
        buildingData = data;
        int finalBlobCount = blobCount;
        int finalChunkCount = chunkCount;
        server.execute(() -> LOGGER.info(
                "Queued base terrain + {} feature blobs ({} block edits across {} chunks) - "
                        + "entering {} new chunk(s)/tick (~{}s)...",
                finalBlobCount, totalQueued, finalChunkCount, WorldConstants.CHUNKS_PER_TICK,
                finalChunkCount / WorldConstants.CHUNKS_PER_TICK / 20));
    }

    // Resolution used to decide "is any blob nearby" - independent of the heightmap grid's
    // own resolution, just coarse enough to keep the lookup set small.
    private static final int COARSE_CELL = 32;

    private static long packCell(int cx, int cz) {
        return ((long) cx << 32) ^ (cz & 0xffffffffL);
    }

    private static Set<Long> coarseCellsNearBlobs(LatentSeasSavedData data) {
        Set<Long> occupied = new HashSet<>();
        for (BlockPos origin : data.allBlobOrigins().values()) {
            int cx = Math.floorDiv(origin.getX(), COARSE_CELL);
            int cz = Math.floorDiv(origin.getZ(), COARSE_CELL);
            // Just the blob's own coarse cell, not a buffer ring around it - a real crash
            // traced to touching too many distinct chunks too fast, so the margin was cut
            // (32x32 instead of the original 96x96) to shrink that footprint directly.
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
                    if (dx * dx + dy * dy + dz * dz > r * r + 1) continue; // rounded, not a cube
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

    // Two throttles are both needed, not one or the other (learned the hard way - dropping
    // this one while adding chunk-entry throttling reproduced the exact save-storm crash it
    // was meant to fix): capping new-chunk entry rate above prevents outrunning the async
    // chunk-load pipeline, and flushing dirty chunks in small increments here prevents them
    // from piling into one catastrophic ChunkMap.processUnloads tick instead.
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

    /** Backend state is memory-only (see api.py's discovered/placed_flags globals) - replay
     *  the mod's durable copy back into it in case the Python process restarted on its own. */
    private static void resyncBackend(LatentSeasSavedData data) {
        for (int idx : data.getDiscovered()) {
            LatentSeasMod.BACKEND.dig(idx);
        }
        for (Map.Entry<Integer, Float> e : data.getFlagged().entrySet()) {
            LatentSeasMod.BACKEND.flag(e.getKey(), e.getValue());
        }
    }
}
