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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/**
 * Utility class for extracting descriptors from detected objects and matching
 * detections across stereo camera pairs.
 */
public class StereoMatcher {
    private static final Logger logger = new Logger(StereoMatcher.class, LogGroup.VisionModule);

    /** Patch size (in pixels) around center for descriptor calculation */
    private static final int DESCRIPTOR_PATCH_SIZE = 16;
    
    /** Maximum descriptor distance to consider a valid match */
    private static final double MAX_DESCRIPTOR_DISTANCE = 0.5;
    
    /** Weight for Y-coordinate difference in matching (penalize vertical misalignment) */
    private static final double Y_DIFFERENCE_WEIGHT = 0.1;
    
    /** Maximum Y-coordinate difference (in pixels) between matched objects */
    private static final double MAX_Y_DIFFERENCE = 50;

    /**
     * Extract descriptor information for a detected object.
     * 
     * @param image The source image (BGR format)
     * @param result The detection result
     * @param classNames List of class names from the detector
     * @param isLeftCamera Whether this is from the left camera
     * @return A StereoDetectedObject with computed descriptors
     */
    public static StereoDetectedObject extractDescriptor(
            Mat image, NeuralNetworkPipeResult result, List<String> classNames, boolean isLeftCamera) {
        
        Rect2d bbox = result.bbox();
        double centerX = bbox.x + bbox.width / 2.0;
        double centerY = bbox.y + bbox.height / 2.0;
        
        String className = (result.classIdx() >= 0 && result.classIdx() < classNames.size())
                ? classNames.get(result.classIdx())
                : "unknown";
        
        // Calculate patch bounds with boundary checking
        int patchHalf = DESCRIPTOR_PATCH_SIZE / 2;
        int x1 = Math.max(0, (int) centerX - patchHalf);
        int y1 = Math.max(0, (int) centerY - patchHalf);
        int x2 = Math.min(image.cols(), (int) centerX + patchHalf);
        int y2 = Math.min(image.rows(), (int) centerY + patchHalf);
        
        // Ensure valid patch dimensions
        if (x2 <= x1 || y2 <= y1) {
            // Return object with zero descriptors if patch is invalid
            return new StereoDetectedObject(
                    bbox, result.classIdx(), className, result.confidence(),
                    centerX, centerY, 0, 0, 0, 0, isLeftCamera);
        }
        
        Rect patchRect = new Rect(x1, y1, x2 - x1, y2 - y1);
        Mat patch = new Mat(image, patchRect);
        
        // Calculate average RGB
        Scalar meanColor = Core.mean(patch);
        double avgB = meanColor.val[0];
        double avgG = meanColor.val[1];
        double avgR = meanColor.val[2];
        
        // Calculate gradient magnitude
        Mat grayPatch = new Mat();
        Imgproc.cvtColor(patch, grayPatch, Imgproc.COLOR_BGR2GRAY);
        
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        Imgproc.Sobel(grayPatch, gradX, org.opencv.core.CvType.CV_64F, 1, 0);
        Imgproc.Sobel(grayPatch, gradY, org.opencv.core.CvType.CV_64F, 0, 1);
        
        Mat magnitude = new Mat();
        Core.magnitude(gradX, gradY, magnitude);
        Scalar meanGradient = Core.mean(magnitude);
        double avgGradient = meanGradient.val[0];
        
        // Release temporary Mats
        patch.release();
        grayPatch.release();
        gradX.release();
        gradY.release();
        magnitude.release();
        
        return new StereoDetectedObject(
                bbox, result.classIdx(), className, result.confidence(),
                centerX, centerY, avgGradient, avgR, avgG, avgB, isLeftCamera);
    }

    /**
     * Match detected objects between left and right camera frames.
     * Uses class matching and descriptor similarity.
     * 
     * @param leftDetections Detections from left camera
     * @param rightDetections Detections from right camera
     * @param config The stereo configuration
     * @return List of matched pairs
     */
    public static List<StereoMatchedPair> matchDetections(
            List<StereoDetectedObject> leftDetections,
            List<StereoDetectedObject> rightDetections,
            StereoConfiguration config) {
        
        List<StereoMatchedPair> matches = new ArrayList<>();
        boolean[] rightUsed = new boolean[rightDetections.size()];
        int matchId = 1;
        
        // Group detections by class for more efficient matching
        Map<Integer, List<Integer>> leftByClass = groupByClass(leftDetections);
        Map<Integer, List<Integer>> rightByClass = groupByClass(rightDetections);
        
        // For each class, match left detections to right detections
        for (Integer classIdx : leftByClass.keySet()) {
            if (!rightByClass.containsKey(classIdx)) {
                continue;
            }
            
            List<Integer> leftIndices = leftByClass.get(classIdx);
            List<Integer> rightIndices = rightByClass.get(classIdx);
            
            // Create candidate matches for this class
            List<CandidateMatch> candidates = new ArrayList<>();
            
            for (int li : leftIndices) {
                StereoDetectedObject leftObj = leftDetections.get(li);
                
                for (int ri : rightIndices) {
                    if (rightUsed[ri]) continue;
                    
                    StereoDetectedObject rightObj = rightDetections.get(ri);
                    
                    // Check Y-coordinate difference (should be similar for rectified stereo)
                    double yDiff = Math.abs(leftObj.centerY - rightObj.centerY);
                    if (yDiff > MAX_Y_DIFFERENCE) continue;
                    
                    // Check that left object is actually to the left (positive disparity)
                    // In stereo, the same object should appear more to the right in the left image
                    double disparity = leftObj.centerX - rightObj.centerX;
                    if (disparity <= 0) continue;  // Object should have positive disparity
                    
                    // Calculate match score
                    double descriptorDist = leftObj.descriptorDistance(rightObj);
                    double score = descriptorDist + yDiff * Y_DIFFERENCE_WEIGHT / MAX_Y_DIFFERENCE;
                    
                    if (descriptorDist < MAX_DESCRIPTOR_DISTANCE) {
                        candidates.add(new CandidateMatch(li, ri, score, descriptorDist));
                    }
                }
            }
            
            // Sort by score and greedily assign matches
            candidates.sort(Comparator.comparingDouble(c -> c.score));
            boolean[] leftUsed = new boolean[leftDetections.size()];
            
            for (CandidateMatch candidate : candidates) {
                if (leftUsed[candidate.leftIdx] || rightUsed[candidate.rightIdx]) {
                    continue;
                }
                
                StereoDetectedObject leftObj = leftDetections.get(candidate.leftIdx);
                StereoDetectedObject rightObj = rightDetections.get(candidate.rightIdx);
                
                // Calculate depth using stereo geometry
                double[] depthInfo = StereoDepthCalculator.calculateDepth(
                        leftObj, rightObj, config);
                
                StereoMatchedPair pair = new StereoMatchedPair(
                        leftObj, rightObj, matchId++,
                        depthInfo[0], depthInfo[1], depthInfo[2],
                        candidate.descriptorDistance);
                
                matches.add(pair);
                leftUsed[candidate.leftIdx] = true;
                rightUsed[candidate.rightIdx] = true;
            }
        }
        
        logger.debug(() -> "Matched " + matches.size() + " stereo pairs from "
                + leftDetections.size() + " left and " + rightDetections.size() + " right detections");
        
        return matches;
    }

    /**
     * Group detection indices by class index.
     */
    private static Map<Integer, List<Integer>> groupByClass(List<StereoDetectedObject> detections) {
        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int i = 0; i < detections.size(); i++) {
            int classIdx = detections.get(i).classIdx;
            groups.computeIfAbsent(classIdx, k -> new ArrayList<>()).add(i);
        }
        return groups;
    }

    /**
     * Helper class for candidate match scoring.
     */
    private static class CandidateMatch {
        final int leftIdx;
        final int rightIdx;
        final double score;
        final double descriptorDistance;
        
        CandidateMatch(int leftIdx, int rightIdx, double score, double descriptorDistance) {
            this.leftIdx = leftIdx;
            this.rightIdx = rightIdx;
            this.score = score;
            this.descriptorDistance = descriptorDistance;
        }
    }
}
