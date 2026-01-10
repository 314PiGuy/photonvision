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
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;
import org.photonvision.vision.pipeline.ObjectDetectionPipeline;
import org.photonvision.vision.pipeline.ObjectDetectionPipelineSettings;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.processes.VisionModule;
import org.photonvision.vision.processes.VisionModuleManager;

/**
 * Manages stereo vision processing for a pair of cameras.
 * Runs object detection on both cameras, matches detections, and calculates depth.
 */
public class StereoVisionModule {
    private static final Logger logger = new Logger(StereoVisionModule.class, LogGroup.VisionModule);

    private final StereoConfiguration config;
    private final AtomicBoolean running = new AtomicBoolean(false);
    
    // Target FPS for stereo processing
    private static final double TARGET_FPS = 15.0;
    private static final long MIN_FRAME_INTERVAL_NS = (long)(1_000_000_000.0 / TARGET_FPS);
    // Much stricter frame sync - only use frames within 200ms of each other to avoid latency
    private static final long MAX_FRAME_AGE_MS = 200; // 200 ms max difference
    
    private VisionModule leftModule;
    private VisionModule rightModule;
    
    private volatile CVPipelineResult leftResult;
    private volatile CVPipelineResult rightResult;
    private volatile Frame leftFrame;
    private volatile Frame rightFrame;
    private volatile long leftFrameTimestampNanos = 0;
    private volatile long rightFrameTimestampNanos = 0;
    
    private final Object resultLock = new Object();
    private long lastProcessTime = System.nanoTime();
    private long lastOutputTime = 0;
    private double currentFps = 0;
    private int processedFrameCount = 0;
    private int skippedFrameCount = 0;
    private long lastStatusLogTime = 0;
    
    private final List<Consumer<StereoResult>> resultConsumers = new ArrayList<>();
    private final List<Consumer<String>> imageConsumers = new ArrayList<>();
    private volatile String latestStereoImage;

    private static Mat selectBestMatForDisplay(Frame frame) {
        if (frame == null) return new Mat();
        Mat color = frame.colorImage != null ? frame.colorImage.getMat() : null;
        if (color != null && !color.empty()) {
            return color.clone();
        }

        Mat processed = frame.processedImage != null ? frame.processedImage.getMat() : null;
        if (processed == null || processed.empty()) {
            return new Mat();
        }

        // processedImage is often thresholded/gray; convert to BGR so drawing + hconcat behave.
        Mat out = new Mat();
        if (processed.channels() == 1) {
            Imgproc.cvtColor(processed, out, Imgproc.COLOR_GRAY2BGR);
        } else {
            out = processed.clone();
        }
        return out;
    }

    private static Mat selectBestMatForDescriptor(Frame frame) {
        if (frame == null) return new Mat();
        Mat color = frame.colorImage != null ? frame.colorImage.getMat() : null;
        if (color != null && !color.empty()) {
            return color;
        }
        Mat processed = frame.processedImage != null ? frame.processedImage.getMat() : null;
        return processed != null ? processed : new Mat();
    }
    
    // Colors for drawing matched pairs
    private static final Scalar[] MATCH_COLORS = {
        new Scalar(255, 0, 0),    // Blue
        new Scalar(0, 255, 0),    // Green
        new Scalar(0, 0, 255),    // Red
        new Scalar(255, 255, 0),  // Cyan
        new Scalar(255, 0, 255),  // Magenta
        new Scalar(0, 255, 255),  // Yellow
        new Scalar(128, 0, 255),  // Purple
        new Scalar(255, 128, 0),  // Orange
    };

    public StereoVisionModule(StereoConfiguration config) {
        this.config = config;
    }

    /**
     * Initialize the stereo module by finding the camera modules.
     * 
     * @param vmm The VisionModuleManager containing all camera modules
     * @return true if both cameras were found
     */
    public boolean initialize(VisionModuleManager vmm) {
        leftModule = vmm.getModule(config.leftCameraName);
        rightModule = vmm.getModule(config.rightCameraName);
        
        if (leftModule == null) {
            logger.error("Left camera not found: " + config.leftCameraName);
            return false;
        }
        if (rightModule == null) {
            logger.error("Right camera not found: " + config.rightCameraName);
            return false;
        }
        
        logger.info("Stereo module initialized with cameras: " 
                + config.leftCameraName + " (left), " + config.rightCameraName + " (right)");
        
        return true;
    }

    /**
     * Start stereo processing.
     */
    public void start() {
        if (running.getAndSet(true)) {
            return;  // Already running
        }
        
        logger.info("Starting stereo vision module for cameras: " 
                + config.leftCameraName + " (left), " + config.rightCameraName + " (right)");
        logger.info("Note: Both cameras must be configured with object detection pipelines for stereo processing to work.");
        
        // Subscribe to pipeline results from both cameras
        leftModule.addResultConsumer(this::onLeftResult);
        rightModule.addResultConsumer(this::onRightResult);
    }

    /**
     * Stop stereo processing.
     */
    public void stop() {
        if (!running.getAndSet(false)) {
            return;  // Already stopped
        }
        
        logger.info("Stopping stereo vision module");
    }

    /**
     * Add a consumer for stereo results.
     */
    public void addResultConsumer(Consumer<StereoResult> consumer) {
        resultConsumers.add(consumer);
    }

    /**
     * Add a consumer for stereo images (base64 encoded).
     */
    public void addImageConsumer(Consumer<String> consumer) {
        imageConsumers.add(consumer);
    }

    /**
     * Get the latest stereo image (base64 encoded).
     */
    public String getLatestStereoImage() {
        return latestStereoImage;
    }

    /**
     * Get the current stereo configuration.
     */
    public StereoConfiguration getConfiguration() {
        return config;
    }

    /**
     * Check if the module is running.
     */
    public boolean isRunning() {
        return running.get();
    }

    private void onLeftResult(CVPipelineResult result) {
        long handlerStartNs = System.nanoTime();
        synchronized (resultLock) {
            long lockAcquiredNs = System.nanoTime();
            this.leftResult = result;
            if (result.inputAndOutputFrame != null) {
                // Deep-copy the frame so OpenCV mats remain valid after the pipeline continues.
                long copyStartNs = System.nanoTime();
                Frame copy = new Frame();
                result.inputAndOutputFrame.copyTo(copy);
                long copyEndNs = System.nanoTime();
                if (this.leftFrame != null) {
                    try {
                        this.leftFrame.release();
                    } catch (Exception ignored) {
                    }
                }
                this.leftFrame = copy;
                this.leftFrameTimestampNanos = result.inputAndOutputFrame.timestampNanos;
                
                long nowNs = System.nanoTime();
                long frameAgeMs = (nowNs - result.inputAndOutputFrame.timestampNanos) / 1_000_000;
                long copyTimeUs = (copyEndNs - copyStartNs) / 1000;
                long lockWaitUs = (lockAcquiredNs - handlerStartNs) / 1000;
                
                if (frameAgeMs > 1000 || copyTimeUs > 10000) {
                    logger.warn(String.format("LEFT: seq=%d, frameAge=%dms, copyTime=%dus, lockWait=%dus",
                        result.sequenceID, frameAgeMs, copyTimeUs, lockWaitUs));
                }
            }
            tryProcessStereo();
        }
    }

    private void onRightResult(CVPipelineResult result) {
        long handlerStartNs = System.nanoTime();
        synchronized (resultLock) {
            long lockAcquiredNs = System.nanoTime();
            this.rightResult = result;
            if (result.inputAndOutputFrame != null) {
                // Deep-copy the frame so OpenCV mats remain valid after the pipeline continues.
                long copyStartNs = System.nanoTime();
                Frame copy = new Frame();
                result.inputAndOutputFrame.copyTo(copy);
                long copyEndNs = System.nanoTime();
                if (this.rightFrame != null) {
                    try {
                        this.rightFrame.release();
                    } catch (Exception ignored) {
                    }
                }
                this.rightFrame = copy;
                this.rightFrameTimestampNanos = result.inputAndOutputFrame.timestampNanos;
                
                long nowNs = System.nanoTime();
                long frameAgeMs = (nowNs - result.inputAndOutputFrame.timestampNanos) / 1_000_000;
                long copyTimeUs = (copyEndNs - copyStartNs) / 1000;
                long lockWaitUs = (lockAcquiredNs - handlerStartNs) / 1000;
                
                if (frameAgeMs > 1000 || copyTimeUs > 10000) {
                    logger.warn(String.format("RIGHT: seq=%d, frameAge=%dms, copyTime=%dus, lockWait=%dus",
                        result.sequenceID, frameAgeMs, copyTimeUs, lockWaitUs));
                }
            }
            tryProcessStereo();
        }
    }

    /**
     * Try to process stereo if we have results from both cameras.
     */
    private void tryProcessStereo() {
        if (!running.get()) {
            return;
        }
        
        // Periodic status logging
        long nowMs = System.currentTimeMillis();
        if (nowMs - lastStatusLogTime > 5000) {
            lastStatusLogTime = nowMs;
            logger.info("Stereo status: leftFrame=" + (leftFrame != null) + 
                    ", rightFrame=" + (rightFrame != null) + 
                    ", leftResult=" + (leftResult != null) + 
                    ", rightResult=" + (rightResult != null) +
                    ", processed=" + processedFrameCount + ", skipped=" + skippedFrameCount);
        }
        
        // We just need frames, not necessarily the results synced
        Frame lFrame = leftFrame;
        Frame rFrame = rightFrame;
        CVPipelineResult left = leftResult;
        CVPipelineResult right = rightResult;
        
        // Check if we have both frames - this is the minimum requirement
        if (lFrame == null || rFrame == null) {
            return;
        }

        long lTs = leftFrameTimestampNanos;
        long rTs = rightFrameTimestampNanos;
        if (lTs > 0 && rTs > 0) {
            long diffMs = Math.abs(lTs - rTs) / 1_000_000;
            if (diffMs > MAX_FRAME_AGE_MS) {
                skippedFrameCount++;
                if (skippedFrameCount % 10 == 1) {
                    logger.warn("Skipping stereo update - frames too far apart (" + diffMs + "ms). This may indicate camera latency or buffering issues. MJPEG cameras should deliver frames in real time.");
                }
                return;
            }
        }
        
        // Check if enough time has passed since last output (rate limiting to TARGET_FPS)
        long now = System.nanoTime();
        if (now - lastOutputTime < MIN_FRAME_INTERVAL_NS) {
            return;  // Rate limit to target FPS
        }
        
        // Use the results we have (they might be null, we'll handle it)
        final CVPipelineResult leftFinal = left;
        final CVPipelineResult rightFinal = right;
        
        // We have frames - process them
        processedFrameCount++;
        if (processedFrameCount % 30 == 1) {
            long frameTimeDiff = Math.abs(leftFrameTimestampNanos - rightFrameTimestampNanos) / 1_000_000;
            int leftTargets = leftFinal != null ? leftFinal.targets.size() : 0;
            int rightTargets = rightFinal != null ? rightFinal.targets.size() : 0;
            logger.info("Processing stereo pair #" + processedFrameCount + 
                    " (frame time diff: " + frameTimeDiff + "ms, FPS: " + 
                    String.format("%.1f", currentFps) + 
                    ", leftTargets: " + leftTargets + 
                    ", rightTargets: " + rightTargets + ")");
        }
        
        // Process the stereo pair
        long startTime = System.nanoTime();
        StereoResult result = processSteroPair(leftFinal, rightFinal, lFrame, rFrame);
        long processingEndTime = System.nanoTime();
        
        // Generate annotated stereo view
        long imageStartTime = System.nanoTime();
        String stereoImage = createAnnotatedStereoView(lFrame, rFrame, result);
        long imageEndTime = System.nanoTime();
        
        long processingMs = (processingEndTime - startTime) / 1_000_000;
        long imageGenMs = (imageEndTime - imageStartTime) / 1_000_000;
        
        if (processingMs > 50 || imageGenMs > 50) {
            logger.warn(String.format("STEREO SLOW: processing=%dms, imageGen=%dms",
                processingMs, imageGenMs));
        }
        
        long endTime = imageEndTime;
        
        // Update timing for rate limiting and FPS calculation
        lastOutputTime = endTime;
        long elapsed = endTime - lastProcessTime;
        lastProcessTime = endTime;
        currentFps = 1_000_000_000.0 / elapsed;
        
        // Don't clobber a previously-good image with an empty frame
        if (stereoImage != null && !stereoImage.isEmpty()) {
            latestStereoImage = stereoImage;
        }
        
        // Notify result consumers
        for (Consumer<StereoResult> consumer : resultConsumers) {
            consumer.accept(result);
        }
        
        // Notify image consumers
        if (stereoImage != null && !stereoImage.isEmpty()) {
            for (Consumer<String> consumer : imageConsumers) {
                consumer.accept(stereoImage);
            }
        }
    }

    /**
     * Process a pair of frames from the stereo cameras.
     */
    private StereoResult processSteroPair(
            CVPipelineResult leftResult,
            CVPipelineResult rightResult,
            Frame leftFrame,
            Frame rightFrame) {
        
        long startTime = System.nanoTime();
        
        Mat leftImage = selectBestMatForDescriptor(leftFrame);
        Mat rightImage = selectBestMatForDescriptor(rightFrame);

        if (leftImage.empty()) logger.info("left empty");
        if (rightImage.empty()) logger.info("right empty");
        
        // if (leftImage.empty() || rightImage.empty()) {
        //     logger.debug("Empty images in processSteroPair");
        //     return StereoResult.empty();
        // }
        
        // Get class names from the pipelines (handle null results)
        List<String> leftClassNames = (leftResult != null && leftResult.objectDetectionClassNames != null) 
                ? leftResult.objectDetectionClassNames : List.of();
        List<String> rightClassNames = (rightResult != null && rightResult.objectDetectionClassNames != null) 
                ? rightResult.objectDetectionClassNames : List.of();
        
        // Extract detections from pipeline results
        List<StereoDetectedObject> leftDetections = new ArrayList<>();
        List<StereoDetectedObject> rightDetections = new ArrayList<>();
        
        // Convert tracked targets to stereo detections (handle null results)
        if (!leftImage.empty() && leftResult != null && leftResult.targets != null) {
            for (var target : leftResult.targets) {
                var bbox = target.getMinAreaRect().boundingRect();
                var rect2d = new org.opencv.core.Rect2d(bbox.x, bbox.y, bbox.width, bbox.height);
                
                var nnResult = new NeuralNetworkPipeResult(rect2d, target.getClassID(), target.getConfidence());
                var detection = StereoMatcher.extractDescriptor(leftImage, nnResult, leftClassNames, true);
                leftDetections.add(detection);
            }
        }
        
        if (!rightImage.empty() && rightResult != null && rightResult.targets != null) {
            for (var target : rightResult.targets) {
                var bbox = target.getMinAreaRect().boundingRect();
                var rect2d = new org.opencv.core.Rect2d(bbox.x, bbox.y, bbox.width, bbox.height);
                
                var nnResult = new NeuralNetworkPipeResult(rect2d, target.getClassID(), target.getConfidence());
                var detection = StereoMatcher.extractDescriptor(rightImage, nnResult, rightClassNames, false);
                rightDetections.add(detection);
            }
        }
        
        // Match detections across cameras
        List<StereoMatchedPair> matches = StereoMatcher.matchDetections(
                leftDetections, rightDetections, config);
        
        long processingTime = System.nanoTime() - startTime;
        
        return new StereoResult(
                System.nanoTime(),
                matches,
                leftDetections.size(),
                rightDetections.size(),
                processingTime,
                currentFps,
                true,
            leftImage.empty() ? 0 : leftImage.cols(),
            leftImage.empty() ? 0 : leftImage.rows(),
            rightImage.empty() ? 0 : rightImage.cols(),
            rightImage.empty() ? 0 : rightImage.rows());
    }

    /**
     * Draw matched pairs on stereo images and create side-by-side view.
     * 
     * @param leftFrame Left camera frame
     * @param rightFrame Right camera frame
     * @param result The stereo result with matched pairs
     * @return Combined side-by-side image as Base64 JPEG
     */
    public String createAnnotatedStereoView(Frame leftFrame, Frame rightFrame, StereoResult result) {
        Mat leftImage = selectBestMatForDisplay(leftFrame);
        Mat rightImage = selectBestMatForDisplay(rightFrame);
        
        // Ensure images have the same dimensions and type before concatenating
        if (leftImage.empty() || rightImage.empty()) {
            leftImage.release();
            rightImage.release();
            return "";
        }
        
        // Resize right image to match left image dimensions if needed
        if (leftImage.rows() != rightImage.rows() || leftImage.cols() != rightImage.cols()) {
            Mat resizedRight = new Mat();
            Imgproc.resize(rightImage, resizedRight, leftImage.size());
            rightImage.release();
            rightImage = resizedRight;
        }
        
        // Ensure same type (convert to same color space if needed)
        if (leftImage.type() != rightImage.type()) {
            Mat convertedRight = new Mat();
            if (leftImage.channels() == 3 && rightImage.channels() == 1) {
                Imgproc.cvtColor(rightImage, convertedRight, Imgproc.COLOR_GRAY2BGR);
            } else if (leftImage.channels() == 1 && rightImage.channels() == 3) {
                Imgproc.cvtColor(rightImage, convertedRight, Imgproc.COLOR_BGR2GRAY);
            } else {
                // If types still don't match, convert both to BGR
                if (leftImage.channels() != 3) {
                    Mat convertedLeft = new Mat();
                    Imgproc.cvtColor(leftImage, convertedLeft, Imgproc.COLOR_GRAY2BGR);
                    leftImage.release();
                    leftImage = convertedLeft;
                }
                if (rightImage.channels() != 3) {
                    Imgproc.cvtColor(rightImage, convertedRight, Imgproc.COLOR_GRAY2BGR);
                } else {
                    convertedRight = rightImage.clone();
                }
            }
            rightImage.release();
            rightImage = convertedRight;
        }
        
        // Draw matched pairs
        for (StereoMatchedPair pair : result.matchedPairs) {
            Scalar color = MATCH_COLORS[pair.matchId % MATCH_COLORS.length];
            
            // Draw on left image
            drawDetection(leftImage, pair.leftDetection, pair.matchId, color);
            
            // Draw on right image
            drawDetection(rightImage, pair.rightDetection, pair.matchId, color);
        }
        
        // Draw unmatched detections (would need to track these separately)
        
        // Create side-by-side view
        Mat combined = new Mat();
        List<Mat> images = new ArrayList<>();
        images.add(leftImage);
        images.add(rightImage);
        Core.hconcat(images, combined);
        
        // Add labels
        Imgproc.putText(combined, "LEFT", new Point(10, 30),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255), 2);
        Imgproc.putText(combined, "RIGHT", new Point(leftImage.cols() + 10, 30),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255), 2);
        
        // Encode to JPEG
        MatOfByte buffer = new MatOfByte();
        MatOfInt params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 80);
        Imgcodecs.imencode(".jpg", combined, buffer, params);
        
        String base64 = Base64.getEncoder().encodeToString(buffer.toArray());
        
        // Cleanup
        leftImage.release();
        rightImage.release();
        combined.release();
        buffer.release();
        params.release();
        
        return base64;
    }

    private void drawDetection(Mat image, StereoDetectedObject detection, int matchId, Scalar color) {
        // Draw bounding box
        Point tl = new Point(detection.boundingBox.x, detection.boundingBox.y);
        Point br = new Point(
                detection.boundingBox.x + detection.boundingBox.width,
                detection.boundingBox.y + detection.boundingBox.height);
        Imgproc.rectangle(image, tl, br, color, 2);
        
        // Draw match ID
        String label = "#" + matchId + " " + detection.className;
        Point labelPos = new Point(tl.x, tl.y - 10);
        Imgproc.putText(image, label, labelPos, 
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        
        // Draw center point
        Point center = new Point(detection.centerX, detection.centerY);
        Imgproc.circle(image, center, 5, color, -1);
    }
}
