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

package org.photonvision.vision.objects;

import java.awt.GraphicsEnvironment;
import java.awt.HeadlessException;
import java.io.IOException;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.OnnxDetectorJNI;
import org.photonvision.onnx.OnnxJNI;
import org.photonvision.onnx.OnnxJNI.OnnxResult;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/** ONNX runtime backed object detector implementation. */
public class OnnxObjectDetector implements ObjectDetector {
    private static final boolean DEBUG_INPUT_FRAME;
    private static final boolean DEBUG_RESULTS;
    private static final Scalar LETTERBOX_COLOR = new Scalar(114.0, 114.0, 114.0);
    private static final Logger logger = new Logger(OnnxDetectorJNI.class, LogGroup.General);
    private static final AtomicBoolean HEADLESS_INPUT_WARNING = new AtomicBoolean(false);
    private static final AtomicBoolean MISSING_INPUT_SIZE_LOGGED = new AtomicBoolean(false);
    private static final AtomicBoolean MISSING_MAX_SCORE_LOGGED = new AtomicBoolean(false);

    static {
        String debugProp = System.getProperty("photon.onnx.debugInput");
        if (debugProp == null) {
            debugProp = System.getenv("PHOTON_ONNX_DEBUG_INPUT");
        }
        DEBUG_INPUT_FRAME = Boolean.parseBoolean(debugProp);

        String debugResultsProp = System.getProperty("photon.onnx.debugDetections");
        if (debugResultsProp == null) {
            debugResultsProp = System.getenv("PHOTON_ONNX_DEBUG_DETECTIONS");
        }
        DEBUG_RESULTS = Boolean.parseBoolean(debugResultsProp);
    }

    private final Cleaner cleaner = Cleaner.create();
    private final Cleaner.Cleanable cleanable;
    private final DetectorState state;
    private final long ptr;
    private final OnnxModel model;
    private final Size inputSize;

    public OnnxObjectDetector(OnnxModel model, Size configuredInputSize) {
        this.model = model;

        try {
            OnnxDetectorJNI.forceLoad();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX native libraries", e);
        }

        long instancePtr;
        try {
            String modelPath = model.modelFile.getPath();
            logger.info("Creating ONNX detector for model at: " + modelPath);
            logger.info("Model file exists: " + model.modelFile.exists());
            logger.info("Model file size: " + model.modelFile.length() + " bytes");
            logger.info("Model file absolute path: " + model.modelFile.getAbsolutePath());
            instancePtr = OnnxJNI.create(modelPath);
        } catch (RuntimeException ex) {
            logger.error("Failed to create ONNX detector from path " + model.modelFile.getPath(), ex);
            throw ex;
        }

        if (instancePtr == 0) {
            throw new RuntimeException(
                    "Failed to create ONNX detector for model " + model.modelFile.getName());
        }

        Size resolvedSize = configuredInputSize;
        try {
            int[] nativeSize = OnnxJNI.getInputSize(instancePtr);
            if (nativeSize != null && nativeSize.length >= 2) {
                int nativeWidth = nativeSize[0];
                int nativeHeight = nativeSize[1];
                if (nativeWidth > 0 && nativeHeight > 0) {
                    resolvedSize = new Size(nativeWidth, nativeHeight);
                }
            }
        } catch (UnsatisfiedLinkError ex) {
            if (MISSING_INPUT_SIZE_LOGGED.compareAndSet(false, true)) {
                logger.debug(
                        "ONNX native library does not expose getInputSize; using configured size "
                                + configuredInputSize);
            }
        } catch (RuntimeException ex) {
            logger.warn("Unable to query ONNX model input size: " + ex.getMessage());
        }

        if (!resolvedSize.equals(configuredInputSize)) {
            logger.info(
                    "Using ONNX model-reported input size "
                            + resolvedSize
                            + " instead of configured size "
                            + configuredInputSize);
        }

        this.inputSize = resolvedSize;
        this.ptr = instancePtr;

        this.state = new DetectorState(ptr, model.modelFile.getName());
        this.cleanable = cleaner.register(state, state);
    }

    @Override
    public OnnxModel getModel() {
        return model;
    }

    @Override
    public List<String> getClasses() {
        return model.properties.labels();
    }

    @Override
    public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
        if (state.isReleased()) {
            logger.warn("Attempted to use ONNX detector after release for model " + model.modelFile.getName());
            return List.of();
        }

        if (ptr == 0) {
            logger.error("ONNX detector is not initialized for model " + model.modelFile.getName());
            return List.of();
        }

        if (in == null || in.empty()) {
            logger.warn("Input frame for ONNX detector was null or empty");
            return List.of();
        }

        Mat converted = null;
        Mat bgrInput = in;

        if (in.channels() == 1) {
            converted = new Mat();
            Imgproc.cvtColor(in, converted, Imgproc.COLOR_GRAY2BGR);
            bgrInput = converted;
        } else if (in.channels() == 4) {
            converted = new Mat();
            Imgproc.cvtColor(in, converted, Imgproc.COLOR_BGRA2BGR);
            bgrInput = converted;
        }

        Mat letterboxed = new Mat();
        Letterbox scale = Letterbox.letterbox(bgrInput, letterboxed, inputSize, LETTERBOX_COLOR);

        if (DEBUG_RESULTS) {
            logger.debug(
                    "Letterbox scale="
                            + scale.scale
                            + " dx="
                            + scale.dx
                            + " dy="
                            + scale.dy
                            + " original="
                            + in.size()
                            + " letterboxed="
                            + letterboxed.size());
        }

        if (converted != null) {
            converted.release();
        }

        if ((int) letterboxed.size().width != (int) inputSize.width
                || (int) letterboxed.size().height != (int) inputSize.height) {
            letterboxed.release();
            throw new RuntimeException(
                    "Letterboxed frame was "
                            + letterboxed.size()
                            + " but expected "
                            + inputSize);
        }

        if (DEBUG_INPUT_FRAME) {
            if (!GraphicsEnvironment.isHeadless()) {
                try {
                    HighGui.imshow("PhotonVision ONNX Input", letterboxed);
                    HighGui.waitKey(1);
                } catch (HeadlessException ex) {
                    if (HEADLESS_INPUT_WARNING.compareAndSet(false, true)) {
                        logger.warn(
                                "Unable to display ONNX debug input frame in headless environment: "
                                        + ex.getMessage());
                    }
                }
            } else if (HEADLESS_INPUT_WARNING.compareAndSet(false, true)) {
                logger.warn("ONNX debug input frame requested but environment is headless; skipping");
            }
        }

    OnnxResult[] results =
                OnnxJNI.detect(
                        ptr,
                        letterboxed.getNativeObjAddr(),
                        boxThresh,
                        nmsThresh,
                        model.properties.labels().size());

    double lastMaxScore = resolveLastMaxScore(results);
    state.recordLastMaxScore(lastMaxScore);

        if (DEBUG_RESULTS) {
            double maxScore = lastMaxScore;
            logger.debug(
                    "ONNX detector returned "
                            + (results == null ? 0 : results.length)
                            + " raw detections for model "
                            + model.modelFile.getName()
                            + ", max raw class score="
                            + (Double.isNaN(maxScore) ? "n/a" : String.format("%.4f", maxScore))
                            + " (threshold="
                            + boxThresh
                            + ")");
            if (results != null) {
                for (int i = 0; i < results.length; i++) {
                    OnnxResult result = results[i];
                    logger.debug(
                            "Raw detection #"
                                    + i
                                    + " class="
                                    + result.classId
                                    + " conf="
                                    + result.confidence
                                    + " rect="
                                    + result.rect);
                }
            }
        }

        letterboxed.release();

        if (results == null || results.length == 0) {
            return List.of();
        }

        List<NeuralNetworkPipeResult> detections =
                scale.resizeDetections(
                        Arrays.stream(results)
                                .map(
                                        result ->
                                                new NeuralNetworkPipeResult(
                                                        result.rect, result.classId, result.confidence))
                                .toList());

        if (DEBUG_RESULTS) {
            logger.debug(
                    "Scaled detections count="
                            + detections.size()
                            + " for model "
                            + model.modelFile.getName());
            for (int i = 0; i < detections.size(); i++) {
                NeuralNetworkPipeResult detection = detections.get(i);
                logger.debug(
                        "Scaled detection #"
                                + i
                                + " class="
                                + detection.classIdx()
                                + " conf="
                                + detection.confidence()
                                + " rect="
                                + detection.bbox());
            }
        }

        return detections;
    }

    @Override
    public void release() {
        if (state.destroy()) {
            cleanable.clean();
        }
    }

    public double getLastMaxScore() {
        return state.getLastMaxScore();
    }

    private double resolveLastMaxScore(OnnxResult[] results) {
        double maxScore = Double.NaN;
        try {
            maxScore = OnnxJNI.getLastMaxScore(ptr);
        } catch (UnsatisfiedLinkError error) {
            maxScore = computeJavaFallbackMax(results);
            if (MISSING_MAX_SCORE_LOGGED.compareAndSet(false, true)) {
                logger.info(
                        "Native library missing getLastMaxScore; using Java fallback for max score debug");
            }
        } catch (RuntimeException ex) {
            logger.debug("Unable to query native max score: " + ex.getMessage());
        }
        return maxScore;
    }

    private static double computeJavaFallbackMax(OnnxResult[] results) {
        if (results == null || results.length == 0) {
            return Double.NaN;
        }

        double max = Double.NEGATIVE_INFINITY;
        for (OnnxResult result : results) {
            if (result != null && result.confidence > max) {
                max = result.confidence;
            }
        }

        return max == Double.NEGATIVE_INFINITY ? Double.NaN : max;
    }

    private static final class DetectorState implements Runnable {
        private final long detectorPtr;
    private final AtomicBoolean released = new AtomicBoolean(false);
        private final String modelName;
    private volatile double lastMaxScore = Double.NaN;

        DetectorState(long detectorPtr, String modelName) {
            this.detectorPtr = detectorPtr;
            this.modelName = modelName;
        }

        boolean destroy() {
            if (released.compareAndSet(false, true)) {
                if (detectorPtr != 0) {
                    OnnxJNI.destroy(detectorPtr);
                    logger.debug("Released ONNX detector for model " + modelName);
                }
                return true;
            }
            return false;
        }

        boolean isReleased() {
            return released.get();
        }

        void recordLastMaxScore(double score) {
            lastMaxScore = score;
        }

        double getLastMaxScore() {
            return lastMaxScore;
        }

        @Override
        public void run() {
            destroy();
        }
    }
}
