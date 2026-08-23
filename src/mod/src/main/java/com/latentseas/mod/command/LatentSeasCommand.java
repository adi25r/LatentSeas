package com.latentseas.mod.command;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.latentseas.mod.LatentSeasMod;
import com.latentseas.mod.data.LatentSeasSavedData;
import com.latentseas.mod.world.NavigationCompass;
import com.latentseas.mod.world.ProbeBeacon;
import com.latentseas.mod.world.WorldConstants;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.logging.LogUtils;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraftforge.event.RegisterCommandsEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * The in-game replacement for the web UI's side panel: /latentseas probe <word> lights up
 * beacon beams over whatever it finds (ProbeBeacon), /latentseas target [text] sets/shows the
 * sentence you're trying to steer generation toward, and /latentseas generate asks the
 * backend to generate from just <BOS> (no seed prompt) with whatever's currently flagged
 * (diamonded/activated blocks) doing all the steering - scored against the target if one's
 * set. These are the pieces block/proximity mechanics alone can't reproduce.
 */
public final class LatentSeasCommand {
    private LatentSeasCommand() {}
    private static final Logger LOGGER = LogUtils.getLogger();

    private static String target = "";

    @SubscribeEvent
    public static void onRegister(RegisterCommandsEvent event) {
        event.getDispatcher().register(Commands.literal("latentseas")
                .then(Commands.literal("probe")
                        .then(Commands.argument("word", StringArgumentType.word())
                                .executes(ctx -> {
                                    probe(ctx.getSource(), StringArgumentType.getString(ctx, "word"));
                                    return 1;
                                })))
                .then(Commands.literal("target")
                        .executes(ctx -> {
                            showTarget(ctx.getSource());
                            return 1;
                        })
                        .then(Commands.argument("text", StringArgumentType.greedyString())
                                .executes(ctx -> {
                                    setTarget(ctx.getSource(), StringArgumentType.getString(ctx, "text"));
                                    return 1;
                                })))
                .then(Commands.literal("generate")
                        .executes(ctx -> {
                            generate(ctx.getSource());
                            return 1;
                        }))
                .then(Commands.literal("clear")
                        .executes(ctx -> {
                            clearBeacons(ctx.getSource());
                            return 1;
                        })));
    }

    private static void clearBeacons(CommandSourceStack source) {
        ServerLevel level = source.getServer().getLevel(WorldConstants.FEATURE_WORLD);
        if (level == null) return;
        ProbeBeacon.clear(level);
        source.sendSystemMessage(Component.literal("probe beacons cleared"));
    }

    public static String currentTarget() {
        return target;
    }

    private static void showTarget(CommandSourceStack source) {
        source.sendSystemMessage(Component.literal(
                target.isEmpty() ? "no target set - /latentseas target <sentence>" : "target: " + target));
    }

    private static void setTarget(CommandSourceStack source, String text) {
        target = text;
        source.sendSystemMessage(Component.literal("target set: " + text));
    }

    private static void probe(CommandSourceStack source, String word) {
        ServerPlayer player;
        try {
            player = source.getPlayerOrException();
        } catch (Exception e) {
            return;
        }
        MinecraftServer server = source.getServer();

        LatentSeasMod.BACKEND.probe(word, 10.0)
                .thenAccept(json -> server.execute(() -> onProbeResult(server, player, word, json)))
                .exceptionally(err -> {
                    LOGGER.error("Probe failed", err);
                    return null;
                });
    }

    private static void onProbeResult(MinecraftServer server, ServerPlayer player, String word, JsonObject json) {
        if (json.has("error")) {
            player.sendSystemMessage(Component.literal("probe error: " + json.get("error").getAsString()));
            return;
        }

        JsonArray features = json.getAsJsonArray("activated_features");
        List<Integer> idxs = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            idxs.add(features.get(i).getAsJsonObject().get("feature_idx").getAsInt());
        }

        ProbeBeacon.show(server, idxs);

        if (idxs.isEmpty()) {
            player.sendSystemMessage(Component.literal("probed \"" + word + "\": nothing activated"));
            return;
        }

        player.sendSystemMessage(Component.literal(
                "probed \"" + word + "\": " + idxs.size() + " hit(s) - follow the beacon beams"));

        // Point a compass at the strongest hit (activated_features is sorted by activation,
        // so index 0 is it) - a beam tells you it's out there somewhere, a compass tells you
        // which way to walk.
        ServerLevel level = server.getLevel(WorldConstants.FEATURE_WORLD);
        BlockPos origin = LatentSeasSavedData.get(server).getBlobOrigin(idxs.get(0));
        if (level != null && origin != null) {
            NavigationCompass.giveTo(player, level, origin);
            player.sendSystemMessage(Component.literal("a compass now points at the strongest hit"));
        }
    }

    private static void generate(CommandSourceStack source) {
        ServerPlayer player;
        try {
            player = source.getPlayerOrException();
        } catch (Exception e) {
            return;
        }
        MinecraftServer server = source.getServer();
        String activeTarget = target.isEmpty() ? null : target;

        player.sendSystemMessage(Component.literal("generating..."));
        // Empty prompt, not user text: HookedTransformer.to_tokens("") still prepends BOS
        // by default, so this feeds the model just <BOS> - generation is driven purely by
        // whatever's flagged (right-clicked/diamond-boosted), not by any seed text.
        LatentSeasMod.BACKEND.generate("", 50, 0.7, activeTarget)
                .thenAccept(json -> server.execute(() -> onGenerateResult(player, json)))
                .exceptionally(err -> {
                    LOGGER.error("Generate failed", err);
                    server.execute(() -> player.sendSystemMessage(Component.literal("generate failed - is the backend running?")));
                    return null;
                });
    }

    private static void onGenerateResult(ServerPlayer player, JsonObject json) {
        if (json.has("error")) {
            player.sendSystemMessage(Component.literal("generate error: " + json.get("error").getAsString()));
            return;
        }

        player.sendSystemMessage(Component.literal("generated: " + json.get("generated").getAsString()));
        if (json.has("flags_used")) {
            player.sendSystemMessage(Component.literal("flags used: " + json.get("flags_used")));
        }
        if (json.has("score") && !json.get("score").isJsonNull()) {
            player.sendSystemMessage(Component.literal(
                    "score vs target: " + json.get("score").getAsDouble()));
        }
    }
}
