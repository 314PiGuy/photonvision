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

package org.photonvision.vision.pipeline;

import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

public abstract class CVPipeline<R extends CVPipelineResult, S extends CVPipelineSettings>
        implements Releasable {
    static final int MAX_MULTI_TARGET_RESULTS = 10;

    protected S settings;
    protected FrameStaticProperties frameStaticProperties;
    protected QuirkyCamera cameraQuirks;

    private final FrameThresholdType thresholdType;

    // So releaseable doesn't keep track of if we double-free something. so (ew) remember that here
    protected volatile boolean released = false;

    public CVPipeline(FrameThresholdType thresholdType) {
        this.thresholdType = thresholdType;
    }

    public FrameThresholdType getThresholdType() {
        return thresholdType;
    }

    protected void setPipeParams(
            FrameStaticProperties frameStaticProperties, S settings, QuirkyCamera cameraQuirks) {
        this.settings = settings;
        this.frameStaticProperties = frameStaticProperties;
        this.cameraQuirks = cameraQuirks;

        setPipeParamsImpl();
    }

    protected abstract void setPipeParamsImpl();

    protected abstract R process(Frame frame, S settings);

    public S getSettings() {
        return settings;
    }

    public void setSettings(S s) {
        this.settings = s;
    }

    private volatile Frame lastDebugFrame = null;

    public R run(Frame frame, QuirkyCamera cameraQuirks) {
        if (released) {
            throw new RuntimeException("Pipeline use-after-free!");
        }
        if (settings == null) {
            throw new RuntimeException("No settings provided for pipeline!");
        }
        setPipeParams(frame.frameStaticProperties, settings, cameraQuirks);

        // if (frame.image.getMat().empty()) {
        //     //noinspection unchecked
        //     return (R) new CVPipelineResult(0, 0, List.of(), frame);
        // }
        R result = process(frame, settings);

        result.setImageCaptureTimestampNanos(frame.timestampNanos);

        // Store a copy of the output frame to be used by the UI for node previews
        try {
            if (result.inputAndOutputFrame != null) {
                updateDebugFrame(result.inputAndOutputFrame);
            }
        } catch (Exception e) {
            // Never allow debug frame failures to break the pipeline
            e.printStackTrace();
        }

        return result;
    }

    protected synchronized void updateDebugFrame(Frame src) {
        if (src == null) return;
        Frame copy = new Frame();
        src.copyTo(copy);
        if (lastDebugFrame != null) {
            try {
                lastDebugFrame.release();
            } catch (Exception ignored) {
            }
        }
        lastDebugFrame = copy;
    }

    /**
     * Return the latest debug image for this pipeline node, encoded as base64 jpeg. Empty if no frame
     * available.
     */
    public java.util.Optional<String> getDebugImageBase64() {
        Frame f = lastDebugFrame;
        if (f == null) return java.util.Optional.empty();

        try {
            var mat = f.processedImage.getMat();
            var buf = new org.opencv.core.MatOfByte();
            boolean ok = org.opencv.imgcodecs.Imgcodecs.imencode(".jpg", mat, buf);
            if (!ok) return java.util.Optional.empty();
            byte[] bytes = buf.toArray();
            String b64 = java.util.Base64.getEncoder().encodeToString(bytes);
            return java.util.Optional.of(b64);
        } catch (Exception e) {
            return java.util.Optional.empty();
        }
    }

    /** Return the pipeline node at the given path. Default: if path is empty or null, return this. */
    public CVPipeline getNodeAtPath(java.util.List<Integer> path) {
        if (path == null || path.isEmpty()) return this;
        return null;
    }

    /**
     * Release any native memory associated with this pipeline. Called by pipelinemanager at pipeline
     * switch. Stubbed out, but override if needed.
     */
    @Override
    public void release() {
        released = true;
    }
}
