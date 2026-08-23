package com.latentseas.mod.net;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * Talks to the LatentSeas FastAPI backend (src/backend/api.py) — the same endpoints
 * src/frontend/app.js called over fetch(). Every method returns a CompletableFuture that
 * completes on an HttpClient worker thread, NOT the server thread: callers must marshal
 * any world/player-touching follow-up back via ServerLifecycleHooks.getCurrentServer()
 * .execute(...) rather than acting on the result directly.
 */
public class BackendClient {
    private final String baseUrl;
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    public BackendClient(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String baseUrl() {
        return baseUrl;
    }

    public CompletableFuture<Boolean> ping() {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/"))
                .timeout(Duration.ofSeconds(5)).GET().build();
        return http.sendAsync(req, HttpResponse.BodyHandlers.discarding())
                .thenApply(res -> res.statusCode() == 200)
                .exceptionally(err -> false);
    }

    /** GET /pointmap?profile=minecraft — the coarser, blob-spaced layout (see backend api.py). */
    public CompletableFuture<JsonObject> getPointmap(String profile) {
        String url = baseUrl + "/pointmap" + (profile == null ? "" : "?profile=" + profile);
        return getJson(url);
    }

    /** POST /probe {sentence, threshold} — single word only, enforced backend-side too. */
    public CompletableFuture<JsonObject> probe(String word, double threshold) {
        JsonObject body = new JsonObject();
        body.addProperty("sentence", word);
        body.addProperty("threshold", threshold);
        return postJson("/probe", body);
    }

    /** POST /dig {feature_idx} */
    public CompletableFuture<JsonObject> dig(int featureIdx) {
        JsonObject body = new JsonObject();
        body.addProperty("feature_idx", featureIdx);
        return postJson("/dig", body);
    }

    /** POST /flag {feature_idx, strength} */
    public CompletableFuture<JsonObject> flag(int featureIdx, double strength) {
        JsonObject body = new JsonObject();
        body.addProperty("feature_idx", featureIdx);
        body.addProperty("strength", strength);
        return postJson("/flag", body);
    }

    /** DELETE /flag/{featureIdx} */
    public CompletableFuture<JsonObject> unflag(int featureIdx) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/flag/" + featureIdx))
                .timeout(Duration.ofSeconds(10)).DELETE().build();
        return send(req);
    }

    /** POST /generate {prompt, max_tokens, temperature, target} */
    public CompletableFuture<JsonObject> generate(String prompt, int maxTokens, double temperature, String target) {
        JsonObject body = new JsonObject();
        body.addProperty("prompt", prompt);
        body.addProperty("max_tokens", maxTokens);
        body.addProperty("temperature", temperature);
        if (target != null) body.addProperty("target", target);
        return postJson("/generate", body);
    }

    private CompletableFuture<JsonObject> getJson(String url) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(30)).GET().build();
        return send(req);
    }

    private CompletableFuture<JsonObject> postJson(String path, JsonObject body) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                .build();
        return send(req);
    }

    private CompletableFuture<JsonObject> send(HttpRequest req) {
        return http.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                .thenApply(res -> JsonParser.parseString(res.body()).getAsJsonObject());
    }
}
