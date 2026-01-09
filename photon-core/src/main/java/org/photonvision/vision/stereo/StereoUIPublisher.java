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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.photonvision.common.dataflow.DataChangeService;
import org.photonvision.common.dataflow.events.OutgoingUIEvent;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * Publisher that sends stereo vision results to the UI via websocket.
 */
public class StereoUIPublisher implements Consumer<StereoVisionManager.StereoUIData> {
    private static final Logger logger = new Logger(StereoUIPublisher.class, LogGroup.VisionModule);
    
    private static StereoUIPublisher instance;
    private volatile boolean enabled = true;
    
    private StereoUIPublisher() {}

    public static synchronized StereoUIPublisher getInstance() {
        if (instance == null) {
            instance = new StereoUIPublisher();
        }
        return instance;
    }

    /**
     * Register this publisher with the StereoVisionManager.
     */
    public void register() {
        StereoVisionManager.getInstance().addUIConsumer(this);
        logger.info("StereoUIPublisher registered");
    }

    /**
     * Unregister this publisher from the StereoVisionManager.
     */
    public void unregister() {
        StereoVisionManager.getInstance().removeUIConsumer(this);
        logger.info("StereoUIPublisher unregistered");
    }

    /**
     * Enable or disable publishing.
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    @Override
    public void accept(StereoVisionManager.StereoUIData data) {
        if (!enabled) return;
        
        try {
            Map<String, Object> payload = convertToMap(data);
            
            // Send via DataChangeService
            DataChangeService.getInstance().publishEvent(
                    new OutgoingUIEvent<>("stereoResult", payload));
        } catch (Exception e) {
            logger.error("Failed to publish stereo result to UI", e);
        }
    }

    private Map<String, Object> convertToMap(StereoVisionManager.StereoUIData data) {
        Map<String, Object> map = new HashMap<>();
        map.put("isValid", data.isValid);
        map.put("fps", data.fps);
        map.put("latencyMs", data.latencyMs);
        map.put("leftDetections", data.leftDetections);
        map.put("rightDetections", data.rightDetections);
        map.put("matchedPairs", data.matchedPairs);
        
        List<Map<String, Object>> targets = new ArrayList<>();
        for (var target : data.targets) {
            Map<String, Object> targetMap = new HashMap<>();
            targetMap.put("matchId", target.matchId);
            targetMap.put("className", target.className);
            targetMap.put("classIdx", target.classIdx);
            targetMap.put("confidence", target.confidence);
            targetMap.put("depth", target.depth);
            targetMap.put("perpendicularDistance", target.perpendicularDistance);
            targetMap.put("verticalOffset", target.verticalOffset);
            targetMap.put("matchQuality", target.matchQuality);
            targets.add(targetMap);
        }
        map.put("targets", targets);
        
        return map;
    }

    /**
     * Publish a stereo image frame to the UI.
     * 
     * @param base64Image The base64 encoded JPEG image
     */
    public void publishStereoImage(String base64Image) {
        if (!enabled || base64Image == null) return;
        
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("stereoImage", base64Image);
            
            DataChangeService.getInstance().publishEvent(
                    new OutgoingUIEvent<>("stereoImage", payload));
        } catch (Exception e) {
            logger.error("Failed to publish stereo image to UI", e);
        }
    }
}
