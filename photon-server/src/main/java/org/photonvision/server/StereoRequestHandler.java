/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.javalin.http.Context;
import io.javalin.http.UploadedFile;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.stereo.StereoConfiguration;
import org.photonvision.vision.stereo.StereoMatchedPair;
import org.photonvision.vision.stereo.StereoResult;
import org.photonvision.vision.stereo.StereoVisionManager;
import org.photonvision.vision.stereo.StereoVisionModule;

/**
 * Handles HTTP requests related to stereo camera functionality.
 */
public class StereoRequestHandler {
    private static final Logger logger = new Logger(StereoRequestHandler.class, LogGroup.WebServer);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Get list of available cameras for stereo pairing.
     */
    public static void onGetAvailableCameras(Context ctx) {
        try {
            List<String> cameras = StereoVisionManager.getInstance().getAvailableCameras();
            ctx.json(cameras);
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error getting available cameras", e);
            ctx.status(500);
            ctx.result("Error getting available cameras: " + e.getMessage());
        }
    }

    /**
     * Get list of active stereo pairs.
     */
    public static void onGetActivePairs(Context ctx) {
        try {
            List<String> pairs = StereoVisionManager.getInstance().getActivePairIds();
            ctx.json(pairs);
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error getting active pairs", e);
            ctx.status(500);
            ctx.result("Error getting active pairs: " + e.getMessage());
        }
    }

    /**
     * Upload and load a stereo configuration JSON file.
     */
    public static void onUploadConfiguration(Context ctx) {
        try {
            UploadedFile file = ctx.uploadedFile("config");
            
            if (file == null) {
                // Try to get JSON from body instead
                String jsonBody = ctx.body();
                if (jsonBody == null || jsonBody.isEmpty()) {
                    ctx.status(400);
                    ctx.result("No configuration provided. Send JSON in body or as 'config' file upload.");
                    return;
                }
                
                String pairId = StereoVisionManager.getInstance().loadConfiguration(jsonBody);
                if (pairId != null) {
                    Map<String, String> response = new HashMap<>();
                    response.put("pairId", pairId);
                    response.put("message", "Stereo pair created successfully");
                    ctx.json(response);
                    ctx.status(200);
                } else {
                    ctx.status(400);
                    ctx.result("Failed to create stereo pair. Check camera names and configuration.");
                }
                return;
            }
            
            // Read file content
            String json = new BufferedReader(
                    new InputStreamReader(file.content(), StandardCharsets.UTF_8))
                    .lines()
                    .collect(Collectors.joining("\n"));
            
            String pairId = StereoVisionManager.getInstance().loadConfiguration(json);
            
            if (pairId != null) {
                Map<String, String> response = new HashMap<>();
                response.put("pairId", pairId);
                response.put("message", "Stereo pair created successfully");
                ctx.json(response);
                ctx.status(200);
            } else {
                ctx.status(400);
                ctx.result("Failed to create stereo pair. Check camera names and configuration.");
            }
        } catch (Exception e) {
            logger.error("Error uploading stereo configuration", e);
            ctx.status(500);
            ctx.result("Error processing configuration: " + e.getMessage());
        }
    }

    /**
     * Remove a stereo pair.
     */
    public static void onRemovePair(Context ctx) {
        try {
            String pairId = ctx.queryParam("pairId");
            
            if (pairId == null || pairId.isEmpty()) {
                ctx.status(400);
                ctx.result("Missing required parameter: pairId");
                return;
            }
            
            StereoVisionManager.getInstance().removeStereoModule(pairId);
            
            ctx.status(200);
            ctx.result("Stereo pair removed: " + pairId);
        } catch (Exception e) {
            logger.error("Error removing stereo pair", e);
            ctx.status(500);
            ctx.result("Error removing stereo pair: " + e.getMessage());
        }
    }

    /**
     * Get the current stereo result for a pair.
     */
    public static void onGetResult(Context ctx) {
        try {
            String pairId = ctx.queryParam("pairId");
            
            StereoVisionManager manager = StereoVisionManager.getInstance();
            StereoResult result;
            
            if (pairId != null && !pairId.isEmpty()) {
                StereoVisionModule module = manager.getModule(pairId);
                if (module == null) {
                    ctx.status(404);
                    ctx.result("Stereo pair not found: " + pairId);
                    return;
                }
                // For now, return the latest global result
                result = manager.getLatestResult();
            } else {
                result = manager.getLatestResult();
            }
            
            if (result == null) {
                ctx.json(StereoResult.empty());
            } else {
                ctx.json(convertResultToMap(result));
            }
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error getting stereo result", e);
            ctx.status(500);
            ctx.result("Error getting stereo result: " + e.getMessage());
        }
    }

    /**
     * Get the configuration for a stereo pair.
     */
    public static void onGetConfiguration(Context ctx) {
        try {
            String pairId = ctx.queryParam("pairId");
            
            if (pairId == null || pairId.isEmpty()) {
                ctx.status(400);
                ctx.result("Missing required parameter: pairId");
                return;
            }
            
            StereoVisionModule module = StereoVisionManager.getInstance().getModule(pairId);
            
            if (module == null) {
                ctx.status(404);
                ctx.result("Stereo pair not found: " + pairId);
                return;
            }
            
            ctx.json(module.getConfiguration());
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error getting stereo configuration", e);
            ctx.status(500);
            ctx.result("Error getting configuration: " + e.getMessage());
        }
    }

    /**
     * Get the annotated stereo view image.
     */
    public static void onGetStereoImage(Context ctx) {
        try {
            String base64Image = StereoVisionManager.getInstance().getLatestStereoImage();
            
            if (base64Image == null || base64Image.isEmpty()) {
                ctx.status(204);  // No content
                ctx.result("");
                return;
            }
            
            Map<String, String> response = new HashMap<>();
            response.put("image", base64Image);
            ctx.json(response);
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error getting stereo image", e);
            ctx.status(500);
            ctx.result("Error getting stereo image: " + e.getMessage());
        }
    }

    /**
     * Create a sample configuration template.
     */
    public static void onGetConfigTemplate(Context ctx) {
        try {
            StereoConfiguration template = new StereoConfiguration(
                    "camera_left",
                    "camera_right",
                    new StereoConfiguration.CameraPosition(-0.15, 0, 0, 0, 0, 0),
                    new StereoConfiguration.CameraPosition(0.15, 0, 0, 0, 0, 0),
                    new StereoConfiguration.CameraFOV(70.0, 50.0),
                    new StereoConfiguration.CameraFOV(70.0, 50.0));
            
            ctx.json(template);
            ctx.status(200);
        } catch (Exception e) {
            logger.error("Error creating config template", e);
            ctx.status(500);
            ctx.result("Error creating config template: " + e.getMessage());
        }
    }

    private static Map<String, Object> convertResultToMap(StereoResult result) {
        Map<String, Object> map = new HashMap<>();
        map.put("isValid", result.isValid);
        map.put("fps", result.fps);
        map.put("processingTimeMs", result.processingTimeNanos / 1_000_000.0);
        map.put("leftDetections", result.leftDetectionCount);
        map.put("rightDetections", result.rightDetectionCount);
        map.put("matchCount", result.getMatchCount());
        
        List<Map<String, Object>> targets = result.matchedPairs.stream()
                .map(StereoRequestHandler::convertPairToMap)
                .collect(Collectors.toList());
        map.put("targets", targets);
        
        return map;
    }

    private static Map<String, Object> convertPairToMap(StereoMatchedPair pair) {
        Map<String, Object> map = new HashMap<>();
        map.put("matchId", pair.matchId);
        map.put("className", pair.getClassName());
        map.put("classIdx", pair.getClassIdx());
        map.put("confidence", pair.averageConfidence);
        map.put("depth", pair.depth);
        map.put("perpendicularDistance", pair.perpendicularDistance);
        map.put("verticalOffset", pair.verticalOffset);
        map.put("matchQuality", pair.matchQuality);
        return map;
    }
}
