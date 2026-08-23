package com.latentseas.mod.world;

import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.entity.SignBlockEntity;
import net.minecraft.world.level.block.entity.SignText;

import java.util.ArrayList;
import java.util.List;

/**
 * A real, readable sign standing on a feature's block once it's discovered - so its label
 * persists in the world itself rather than only ever having shown up as a chat line you can
 * scroll past.
 */
final class FeatureSign {
    private FeatureSign() {}

    private static final int MAX_LINE_LENGTH = 15;
    private static final int MAX_LINES = 4;

    static void place(ServerLevel level, BlockPos featurePos, String label) {
        BlockPos signPos = featurePos.above();
        level.setBlock(signPos, Blocks.OAK_SIGN.defaultBlockState(), Block.UPDATE_CLIENTS);

        BlockEntity be = level.getBlockEntity(signPos);
        if (!(be instanceof SignBlockEntity sign)) return;

        SignText text = sign.getText(true);
        List<String> lines = wrap(label);
        for (int i = 0; i < MAX_LINES; i++) {
            String line = i < lines.size() ? lines.get(i) : "";
            text = text.setMessage(i, Component.literal(line));
        }
        sign.setText(text, true);
        sign.setChanged();
        level.sendBlockUpdated(signPos, sign.getBlockState(), sign.getBlockState(), Block.UPDATE_CLIENTS);
    }

    private static List<String> wrap(String label) {
        List<String> lines = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (String word : label.split("\\s+")) {
            if (cur.length() > 0 && cur.length() + 1 + word.length() > MAX_LINE_LENGTH) {
                lines.add(cur.toString());
                cur = new StringBuilder();
                if (lines.size() == MAX_LINES) return lines;
            }
            if (cur.length() > 0) cur.append(' ');
            cur.append(word);
        }
        if (cur.length() > 0 && lines.size() < MAX_LINES) lines.add(cur.toString());
        return lines;
    }
}
