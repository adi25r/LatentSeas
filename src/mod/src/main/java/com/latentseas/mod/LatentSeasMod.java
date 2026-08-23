package com.latentseas.mod;

import com.latentseas.mod.command.LatentSeasCommand;
import com.latentseas.mod.net.BackendClient;
import com.latentseas.mod.world.BlockTrackingHandler;
import com.latentseas.mod.world.PlayerGateway;
import com.latentseas.mod.world.RevealHandler;
import com.latentseas.mod.world.WorldBuilder;
import com.mojang.logging.LogUtils;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.slf4j.Logger;

/**
 * Entry point. The backend (src/backend/api.py) is the brain — this mod is a thin client
 * that renders its feature layout as terrain and reports block edits back to it, mirroring
 * what src/frontend/app.js did over HTTP before this mod replaced it.
 */
@Mod(LatentSeasMod.MODID)
public class LatentSeasMod {
    public static final String MODID = "latentseas";
    private static final Logger LOGGER = LogUtils.getLogger();

    // 127.0.0.1, not "localhost": uvicorn binds IPv4-only (--host 0.0.0.0), but "localhost"
    // resolves to the IPv6 loopback (::1) first on this machine, which refuses the
    // connection instantly rather than falling back to IPv4.
    public static final BackendClient BACKEND = new BackendClient("http://127.0.0.1:8000");

    public LatentSeasMod(FMLJavaModLoadingContext context) {
        MinecraftForge.EVENT_BUS.register(this);
        MinecraftForge.EVENT_BUS.register(WorldBuilder.class);
        MinecraftForge.EVENT_BUS.register(BlockTrackingHandler.class);
        MinecraftForge.EVENT_BUS.register(PlayerGateway.class);
        MinecraftForge.EVENT_BUS.register(RevealHandler.class);
        MinecraftForge.EVENT_BUS.register(LatentSeasCommand.class);
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {
        LOGGER.info("LatentSeas starting up, checking backend at {}", BACKEND.baseUrl());
        BACKEND.ping().thenAccept(ok -> {
            if (ok) {
                LOGGER.info("Backend reachable.");
                event.getServer().execute(() -> WorldBuilder.begin(event.getServer()));
            } else {
                LOGGER.warn("Backend NOT reachable — start it with ./run_backend.sh before playing.");
            }
        });
    }
}
