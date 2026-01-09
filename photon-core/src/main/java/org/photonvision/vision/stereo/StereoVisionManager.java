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

package org.photonvision.vision.stereo;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import org.photonvision.common.configuration.PathManager;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.processes.VisionModuleManager;

/**
 * Manages all stereo camera pairs in the system.
 * Handles loading configurations, creating stereo modules, and broadcasting results.
 */
public class StereoVisionManager {
    private static final Logger logger = new Logger(StereoVisionManager.class, LogGroup.VisionModule);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    private static StereoVisionManager instance;
    
    private final Map<String, StereoVisionModule> stereoModules = new HashMap<>();
    private final List<Consumer<StereoUIData>> uiConsumers = new CopyOnWriteArrayList<>();
    
    private VisionModuleManager vmm;
    private volatile StereoResult latestResult;
    private volatile String latestStereoImage;
    
    private StereoVisionManager() {}

    public static synchronized StereoVisionManager getInstance() {
        if (instance == null) {
            instance = new StereoVisionManager();
        }
        return instance;
    }

    /**
     * Initialize the manager with the VisionModuleManager.
     */
    public void initialize(VisionModuleManager vmm) {
        this.vmm = vmm;
        logger.info("StereoVisionManager initialized");
        
        // Load any saved stereo configurations
        loadSavedConfigurations();
    }

    /**
     * Load a stereo configuration from a JSON string.
     * 
     * @param jsonConfig The JSON configuration string
     * @return The stereo pair ID if successful, null otherwise
     */
    public String loadConfiguration(String jsonConfig) {
        try {
            StereoConfiguration config = objectMapper.readValue(jsonConfig, StereoConfiguration.class);
            return addStereoModule(config);
        } catch (Exception e) {
            logger.error("Failed to parse stereo configuration", e);
            return null;
        }
    }

    /**
     * Load a stereo configuration from a file.
     * 
     * @param configFile The configuration file
     * @return The stereo pair ID if successful, null otherwise
     */
    public String loadConfigurationFromFile(File configFile) {
        try {
            String json = Files.readString(configFile.toPath());
            return loadConfiguration(json);
        } catch (IOException e) {
            logger.error("Failed to read stereo configuration file: " + configFile, e);
            return null;
        }
    }

    /**
     * Add a new stereo module with the given configuration.
     * 
     * @param config The stereo configuration
     * @return The stereo pair ID if successful, null otherwise
     */
    public String addStereoModule(StereoConfiguration config) {
        if (vmm == null) {
            logger.error("VisionModuleManager not initialized");
            return null;
        }
        
        String pairId = generatePairId(config);
        
        if (stereoModules.containsKey(pairId)) {
            logger.warn("Stereo pair already exists: " + pairId);
            // Stop and remove existing
            StereoVisionModule existing = stereoModules.get(pairId);
            existing.stop();
            stereoModules.remove(pairId);
        }
        
        StereoVisionModule module = new StereoVisionModule(config);
        
        if (!module.initialize(vmm)) {
            logger.error("Failed to initialize stereo module: " + pairId);
            return null;
        }
        
        // Add result consumer to broadcast to UI
        module.addResultConsumer(this::onStereoResult);
        
        // Add image consumer to capture the stereo view
        module.addImageConsumer(this::onStereoImage);
        
        stereoModules.put(pairId, module);
        module.start();
        
        logger.info("Added stereo module: " + pairId);
        
        // Save configuration
        saveConfiguration(pairId, config);
        
        return pairId;
    }

    /**
     * Remove a stereo module.
     * 
     * @param pairId The stereo pair ID to remove
     */
    public void removeStereoModule(String pairId) {
        StereoVisionModule module = stereoModules.remove(pairId);
        if (module != null) {
            module.stop();
            logger.info("Removed stereo module: " + pairId);
            deleteConfiguration(pairId);
        }
    }

    /**
     * Get all active stereo pair IDs.
     */
    public List<String> getActivePairIds() {
        return new ArrayList<>(stereoModules.keySet());
    }

    /**
     * Get a stereo module by ID.
     */
    public StereoVisionModule getModule(String pairId) {
        return stereoModules.get(pairId);
    }

    /**
     * Get the latest stereo result.
     */
    public StereoResult getLatestResult() {
        return latestResult;
    }

    /**
     * Get the latest stereo view image (base64 encoded).
     */
    public String getLatestStereoImage() {
        return latestStereoImage;
    }

    /**
     * Add a consumer for UI updates.
     */
    public void addUIConsumer(Consumer<StereoUIData> consumer) {
        uiConsumers.add(consumer);
    }

    /**
     * Remove a UI consumer.
     */
    public void removeUIConsumer(Consumer<StereoUIData> consumer) {
        uiConsumers.remove(consumer);
    }

    /**
     * Get list of available cameras for stereo pairing.
     */
    public List<String> getAvailableCameras() {
        if (vmm == null) return List.of();
        
        List<String> cameras = new ArrayList<>();
        for (var module : vmm.getModules()) {
            cameras.add(module.getStateAsCameraConfig().uniqueName);
        }
        return cameras;
    }

    private void onStereoResult(StereoResult result) {
        latestResult = result;
        
        // Broadcast to UI consumers
        StereoUIData uiData = createUIData(result);
        for (Consumer<StereoUIData> consumer : uiConsumers) {
            try {
                consumer.accept(uiData);
            } catch (Exception e) {
                logger.error("Error in stereo UI consumer", e);
            }
        }
    }

    private void onStereoImage(String base64Image) {
        latestStereoImage = base64Image;
    }

    private StereoUIData createUIData(StereoResult result) {
        List<StereoUIData.TargetData> targets = new ArrayList<>();
        
        for (StereoMatchedPair pair : result.matchedPairs) {
            targets.add(new StereoUIData.TargetData(
                    pair.matchId,
                    pair.getClassName(),
                    pair.getClassIdx(),
                    pair.averageConfidence,
                    pair.depth,
                    pair.perpendicularDistance,
                    pair.verticalOffset,
                    pair.matchQuality));
        }
        
        return new StereoUIData(
                result.isValid,
                result.fps,
                result.processingTimeNanos / 1_000_000.0,
                result.leftDetectionCount,
                result.rightDetectionCount,
                result.getMatchCount(),
                targets);
    }

    private String generatePairId(StereoConfiguration config) {
        return config.leftCameraName + "_" + config.rightCameraName;
    }

    private void saveConfiguration(String pairId, StereoConfiguration config) {
        try {
            Path stereoConfigDir = PathManager.getInstance().getRootFolder().resolve("stereo");
            Files.createDirectories(stereoConfigDir);
            
            Path configFile = stereoConfigDir.resolve(pairId + ".json");
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(configFile.toFile(), config);
            
            logger.info("Saved stereo configuration: " + configFile);
        } catch (IOException e) {
            logger.error("Failed to save stereo configuration", e);
        }
    }

    private void deleteConfiguration(String pairId) {
        try {
            Path configFile = PathManager.getInstance()
                    .getRootFolder()
                    .resolve("stereo")
                    .resolve(pairId + ".json");
            Files.deleteIfExists(configFile);
        } catch (IOException e) {
            logger.error("Failed to delete stereo configuration", e);
        }
    }

    private void loadSavedConfigurations() {
        try {
            Path stereoConfigDir = PathManager.getInstance().getRootFolder().resolve("stereo");
            if (!Files.exists(stereoConfigDir)) {
                return;
            }
            
            Files.list(stereoConfigDir)
                    .filter(p -> p.toString().endsWith(".json"))
                    .forEach(configFile -> {
                        try {
                            StereoConfiguration config = objectMapper.readValue(
                                    configFile.toFile(), StereoConfiguration.class);
                            addStereoModule(config);
                        } catch (IOException e) {
                            logger.error("Failed to load stereo config: " + configFile, e);
                        }
                    });
        } catch (IOException e) {
            logger.error("Failed to load saved stereo configurations", e);
        }
    }

    /**
     * Stop all stereo modules.
     */
    public void shutdown() {
        for (StereoVisionModule module : stereoModules.values()) {
            module.stop();
        }
        stereoModules.clear();
        logger.info("StereoVisionManager shut down");
    }

    /**
     * Data class for UI updates.
     */
    public static class StereoUIData {
        public final boolean isValid;
        public final double fps;
        public final double latencyMs;
        public final int leftDetections;
        public final int rightDetections;
        public final int matchedPairs;
        public final List<TargetData> targets;

        public StereoUIData(
                boolean isValid,
                double fps,
                double latencyMs,
                int leftDetections,
                int rightDetections,
                int matchedPairs,
                List<TargetData> targets) {
            this.isValid = isValid;
            this.fps = fps;
            this.latencyMs = latencyMs;
            this.leftDetections = leftDetections;
            this.rightDetections = rightDetections;
            this.matchedPairs = matchedPairs;
            this.targets = targets;
        }

        public static class TargetData {
            public final int matchId;
            public final String className;
            public final int classIdx;
            public final double confidence;
            public final double depth;
            public final double perpendicularDistance;
            public final double verticalOffset;
            public final double matchQuality;

            public TargetData(
                    int matchId,
                    String className,
                    int classIdx,
                    double confidence,
                    double depth,
                    double perpendicularDistance,
                    double verticalOffset,
                    double matchQuality) {
                this.matchId = matchId;
                this.className = className;
                this.classIdx = classIdx;
                this.confidence = confidence;
                this.depth = depth;
                this.perpendicularDistance = perpendicularDistance;
                this.verticalOffset = verticalOffset;
                this.matchQuality = matchQuality;
            }
        }
    }
}
