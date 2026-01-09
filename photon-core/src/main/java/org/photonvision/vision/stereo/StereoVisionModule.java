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
    
    private VisionModule leftModule;
    private VisionModule rightModule;
    
    private volatile CVPipelineResult leftResult;
    private volatile CVPipelineResult rightResult;
    private volatile Frame leftFrame;
    private volatile Frame rightFrame;
    
    private final Object resultLock = new Object();
    private long lastProcessTime = System.nanoTime();
    private double currentFps = 0;
    
    private final List<Consumer<StereoResult>> resultConsumers = new ArrayList<>();
    private final List<Consumer<String>> imageConsumers = new ArrayList<>();
    private volatile String latestStereoImage;
    
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
        synchronized (resultLock) {
            logger.debug(() -> "Left camera result received - seq: " + result.sequenceID + 
                    ", hasFrame: " + (result.inputAndOutputFrame != null) +
                    ", targets: " + result.targets.size());
            this.leftResult = result;
            if (result.inputAndOutputFrame != null) {
                this.leftFrame = result.inputAndOutputFrame;
            }
            tryProcessStereo();
        }
    }

    private void onRightResult(CVPipelineResult result) {
        synchronized (resultLock) {
            logger.debug(() -> "Right camera result received - seq: " + result.sequenceID + 
                    ", hasFrame: " + (result.inputAndOutputFrame != null) +
                    ", targets: " + result.targets.size());
            this.rightResult = result;
            if (result.inputAndOutputFrame != null) {
                this.rightFrame = result.inputAndOutputFrame;
            }
            tryProcessStereo();
        }
    }

    /**
     * Try to process stereo if we have results from both cameras.
     */
    private void tryProcessStereo() {
        if (!running.get()) {
            logger.debug("Not processing - module not running");
            return;
        }
        
        CVPipelineResult left = leftResult;
        CVPipelineResult right = rightResult;
        Frame lFrame = leftFrame;
        Frame rFrame = rightFrame;
        
        if (left == null || right == null || lFrame == null || rFrame == null) {
            logger.debug(() -> "Not processing - missing data: left=" + (left != null) + 
                    ", right=" + (right != null) + ", lFrame=" + (lFrame != null) + 
                    ", rFrame=" + (rFrame != null));
            return;
        }
        
        // Just use the most recent frames from each camera - no strict sync required
        // In real stereo systems, cameras would be hardware-synchronized, but for
        // software-based systems we just match the most recent frames
        long leftSeq = left.sequenceID;
        long rightSeq = right.sequenceID;
        
        logger.debug("Processing stereo pair - leftSeq: " + leftSeq + ", rightSeq: " + rightSeq + 
                ", leftTargets: " + left.targets.size() + ", rightTargets: " + right.targets.size());
        
        // Clear results to wait for next pair
        leftResult = null;
        rightResult = null;
        
        // Process the stereo pair
        long startTime = System.nanoTime();
        StereoResult result = processSteroPair(left, right, lFrame, rFrame);
        long endTime = System.nanoTime();
        
        // Calculate FPS
        long elapsed = endTime - lastProcessTime;
        lastProcessTime = endTime;
        currentFps = 1_000_000_000.0 / elapsed;
        
        // Generate annotated stereo view
        String stereoImage = createAnnotatedStereoView(lFrame, rFrame, result);
        latestStereoImage = stereoImage;
        
        // Notify result consumers
        for (Consumer<StereoResult> consumer : resultConsumers) {
            consumer.accept(result);
        }
        
        // Notify image consumers
        for (Consumer<String> consumer : imageConsumers) {
            consumer.accept(stereoImage);
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
        
        Mat leftImage = leftFrame.colorImage.getMat();
        Mat rightImage = rightFrame.colorImage.getMat();
        
        if (leftImage.empty() || rightImage.empty()) {
            return StereoResult.empty();
        }
        
        // Get class names from the pipelines
        List<String> leftClassNames = leftResult.objectDetectionClassNames != null 
                ? leftResult.objectDetectionClassNames : List.of();
        List<String> rightClassNames = rightResult.objectDetectionClassNames != null 
                ? rightResult.objectDetectionClassNames : List.of();
        
        // Extract detections from pipeline results
        // Note: We need to access the raw neural network results
        // For now, we'll work with the tracked targets which have the detection info
        List<StereoDetectedObject> leftDetections = new ArrayList<>();
        List<StereoDetectedObject> rightDetections = new ArrayList<>();
        
        // Convert tracked targets to stereo detections
        for (var target : leftResult.targets) {
            var bbox = target.getMinAreaRect().boundingRect();
            var rect2d = new org.opencv.core.Rect2d(bbox.x, bbox.y, bbox.width, bbox.height);
            
            // Create a mock NeuralNetworkPipeResult for descriptor extraction
            var nnResult = new NeuralNetworkPipeResult(rect2d, target.getClassID(), target.getConfidence());
            var detection = StereoMatcher.extractDescriptor(leftImage, nnResult, leftClassNames, true);
            leftDetections.add(detection);
        }
        
        for (var target : rightResult.targets) {
            var bbox = target.getMinAreaRect().boundingRect();
            var rect2d = new org.opencv.core.Rect2d(bbox.x, bbox.y, bbox.width, bbox.height);
            
            var nnResult = new NeuralNetworkPipeResult(rect2d, target.getClassID(), target.getConfidence());
            var detection = StereoMatcher.extractDescriptor(rightImage, nnResult, rightClassNames, false);
            rightDetections.add(detection);
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
                leftImage.cols(),
                leftImage.rows(),
                rightImage.cols(),
                rightImage.rows());
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
        Mat leftImage = leftFrame.colorImage.getMat().clone();
        Mat rightImage = rightFrame.colorImage.getMat().clone();
        
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
