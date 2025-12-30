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

package org.photonvision.vision.pipe.impl;

import java.util.List;
import java.util.Optional;
import org.opencv.core.Mat;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class ObjectDetectionPipe
        extends CVPipe<
                CVMat, List<NeuralNetworkPipeResult>, ObjectDetectionPipe.ObjectDetectionPipeParams>
        implements Releasable {
    private ObjectDetector detector;
    // Track the UID of the currently-loaded model to avoid re-loading the same model
    private volatile String currentModelUID;

    private static final org.photonvision.common.logging.Logger logger =
            new org.photonvision.common.logging.Logger(ObjectDetectionPipe.class, org.photonvision.common.logging.LogGroup.VisionModule);

    // Global per-UID load counters to detect pathological repeated creations
    private static final java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.atomic.AtomicInteger> loadCounts = new java.util.concurrent.ConcurrentHashMap<>();

    // Timestamp (ms) of last load attempt per UID to implement a cooldown
    private static final java.util.concurrent.ConcurrentHashMap<String, Long> lastLoadMs = new java.util.concurrent.ConcurrentHashMap<>();

    // Threshold beyond which we will log a warning about frequent loads
    private static final int LOAD_WARNING_THRESHOLD = 5;
    // Cooldown in milliseconds to avoid repeated load spam for the same UID
    private static final long LOAD_COOLDOWN_MS = 10_000L;

    public ObjectDetectionPipe() {
        // Avoid eagerly loading the default model here because ONNX/RKNN detectors
        // allocate native resources. Defer creating detectors until params.model() is set
        // in the process method to prevent repeated allocations during startup.
        detector = NullModel.getInstance();
        currentModelUID = NullModel.getInstance().getUID();
    }

    @Override
    protected List<NeuralNetworkPipeResult> process(CVMat in) {
        // Determine desired model UID
        String desiredModelUID = NullModel.getInstance().getUID();
        if (params != null && params.model() != null) {
            try {
                desiredModelUID = params.model().getUID();
            } catch (Exception ignored) {
            }
        }

        // Only load a new detector when the UID actually changes. Synchronize to avoid races
        if (!desiredModelUID.equals(currentModelUID)) {
            synchronized (this) {
                if (!desiredModelUID.equals(currentModelUID)) {
                    final String dUID = desiredModelUID;
                    final String prevUID = currentModelUID;
                    logger.debug(() -> "Loading detector for UID: " + dUID + " (was: " + prevUID + ")");
                    try {
                        detector.release();
                    } catch (Exception ignored) {
                    }
                    // Defensive: if params.model() is null fallback to NullModel
                    try {
                        detector = params != null && params.model() != null ? params.model().load() : NullModel.getInstance();
                    } catch (Exception e) {
                        // If loading the model fails, fall back to NullModel and keep going
                        logger.error("Failed to load detector for UID " + desiredModelUID + ", falling back to NullModel", e);
                        detector = NullModel.getInstance();
                    }
                    currentModelUID = desiredModelUID;

                    // Track load count for diagnostics
                    var counter = loadCounts.computeIfAbsent(desiredModelUID, k -> new java.util.concurrent.atomic.AtomicInteger());
                    int cnt = counter.incrementAndGet();
                    long now = System.currentTimeMillis();
                    long last = lastLoadMs.getOrDefault(desiredModelUID, 0L);

                    if (cnt > LOAD_WARNING_THRESHOLD && (now - last) < LOAD_COOLDOWN_MS) {
                        logger.warn(
                                "Detector for UID "
                                        + desiredModelUID
                                        + " has been created "
                                        + cnt
                                        + " times; further attempts will be suppressed for "
                                        + LOAD_COOLDOWN_MS
                                        + " ms");
                        // Capture a short stack trace to help locate the caller causing repeated loads
                        try {
                            StackTraceElement[] trace = Thread.currentThread().getStackTrace();
                            StringBuilder sb = new StringBuilder();
                            int start = 3; // skip getStackTrace, this method, and lambda machinery
                            int end = Math.min(trace.length, start + 12);
                            for (int i = start; i < end; i++) {
                                sb.append("\n\tat ").append(trace[i].toString());
                            }
                            logger.warn("Load triggered from thread '" + Thread.currentThread().getName() + "' with trace:" + sb.toString());
                        } catch (Exception ignored) {
                        }
                        // Suppress further load attempts for the cooldown period by falling back to NullModel
                        detector = NullModel.getInstance();
                        currentModelUID = desiredModelUID;
                    } else {
                        // record the timestamp of this successful/attempted load
                        lastLoadMs.put(desiredModelUID, now);
                        if (cnt > LOAD_WARNING_THRESHOLD) {
                            logger.warn(
                                    "Detector for UID "
                                            + desiredModelUID
                                            + " has been created "
                                            + cnt
                                            + " times; this may indicate a problem");
                            try {
                                StackTraceElement[] trace = Thread.currentThread().getStackTrace();
                                StringBuilder sb = new StringBuilder();
                                int start = 3;
                                int end = Math.min(trace.length, start + 6);
                                for (int i = start; i < end; i++) {
                                    sb.append("\n\tat ").append(trace[i].toString());
                                }
                                logger.warn("Recent load trace (first frames):\n" + sb.toString());
                            } catch (Exception ignored) {
                            }
                        }
                    }
                }
            }
        }

        Mat frame = in.getMat();
        if (frame.empty()) {
            return List.of();
        }

        return detector.detect(in.getMat(), params.nms(), params.confidence());
    }

    public static record ObjectDetectionPipeParams(double confidence, double nms, Model model) {}

    public List<String> getClassNames() {
        return detector.getClasses();
    }

    @Override
    public void release() {
        detector.release();
    }
}
