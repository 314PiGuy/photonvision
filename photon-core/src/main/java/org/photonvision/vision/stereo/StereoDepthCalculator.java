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

import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * Calculates depth and 3D position from stereo disparity.
 * Uses triangulation based on camera configuration and detected object positions.
 */
public class StereoDepthCalculator {
    private static final Logger logger = new Logger(StereoDepthCalculator.class, LogGroup.VisionModule);

    /** Minimum disparity to calculate valid depth */
    private static final double MIN_DISPARITY = 1.0;
    
    /** Maximum reasonable depth (meters) to filter outliers */
    private static final double MAX_DEPTH = 50.0;

    /**
     * Calculate depth and position from matched stereo detections.
     * 
     * @param leftDetection The detection from the left camera
     * @param rightDetection The detection from the right camera
     * @param config The stereo camera configuration
     * @return Array of [depth, perpendicularDistance, verticalOffset] in meters
     */
    public static double[] calculateDepth(
            StereoDetectedObject leftDetection,
            StereoDetectedObject rightDetection,
            StereoConfiguration config) {
        
        // Calculate disparity (difference in X position between left and right images)
        // In a standard stereo setup, the left camera sees objects shifted to the right
        double disparity = leftDetection.centerX - rightDetection.centerX;
        
        if (disparity < MIN_DISPARITY) {
            logger.debug(() -> "Disparity too small: " + disparity);
            return new double[] {Double.POSITIVE_INFINITY, 0, 0};
        }
        
        // Get baseline (distance between cameras)
        double baseline = config.getBaseline();
        
        // Estimate focal length in pixels from FOV
        // Assuming image width is roughly the same as detection range
        // This is a simplification - in production, use actual camera calibration
        double avgHorizontalFOV = (config.leftFOV.horizontalFOV + config.rightFOV.horizontalFOV) / 2.0;
        
        // Estimate image width from typical detection center positions
        // For better accuracy, this should use actual image dimensions
        double estimatedImageWidth = Math.max(
                leftDetection.boundingBox.x + leftDetection.boundingBox.width,
                rightDetection.boundingBox.x + rightDetection.boundingBox.width) * 1.5;
        if (estimatedImageWidth < 640) estimatedImageWidth = 640;  // Minimum assumption
        
        // Calculate focal length in pixels: f = width / (2 * tan(fov/2))
        double focalLengthPixels = estimatedImageWidth / (2.0 * Math.tan(Math.toRadians(avgHorizontalFOV / 2.0)));
        
        // Calculate depth using stereo geometry: Z = (baseline * focal_length) / disparity
        double rawDepth = (baseline * focalLengthPixels) / disparity;
        
        // Clamp depth to reasonable range
        double depth;
        if (rawDepth > MAX_DEPTH || rawDepth < 0) {
            final double logDepth = rawDepth;
            logger.debug(() -> "Depth out of range: " + logDepth);
            depth = MAX_DEPTH;
        } else {
            depth = rawDepth;
        }
        
        // Calculate perpendicular distance (X offset from center line)
        // Average the X positions from both cameras and convert to world coordinates
        double avgCenterX = (leftDetection.centerX + rightDetection.centerX) / 2.0;
        double imageCenterX = estimatedImageWidth / 2.0;
        double xOffsetPixels = avgCenterX - imageCenterX;
        double perpendicularDistance = (xOffsetPixels * depth) / focalLengthPixels;
        
        // Calculate vertical offset (Y offset from center)
        double avgVerticalFOV = (config.leftFOV.verticalFOV + config.rightFOV.verticalFOV) / 2.0;
        double estimatedImageHeight = estimatedImageWidth * avgVerticalFOV / avgHorizontalFOV;
        double focalLengthVertical = estimatedImageHeight / (2.0 * Math.tan(Math.toRadians(avgVerticalFOV / 2.0)));
        
        double avgCenterY = (leftDetection.centerY + rightDetection.centerY) / 2.0;
        double imageCenterY = estimatedImageHeight / 2.0;
        double yOffsetPixels = avgCenterY - imageCenterY;
        double verticalOffset = (yOffsetPixels * depth) / focalLengthVertical;
        
        final double logDisparity = disparity;
        final double logDepth = depth;
        final double logPerpDist = perpendicularDistance;
        final double logVertOffset = verticalOffset;
        logger.debug(() -> String.format(
                "Stereo depth: disparity=%.1fpx, depth=%.2fm, perpDist=%.2fm, vertOffset=%.2fm",
                logDisparity, logDepth, logPerpDist, logVertOffset));
        
        return new double[] {depth, perpendicularDistance, verticalOffset};
    }

    /**
     * Calculate depth with known image dimensions (more accurate).
     * 
     * @param leftDetection The detection from the left camera
     * @param rightDetection The detection from the right camera
     * @param config The stereo camera configuration
     * @param imageWidth Actual image width in pixels
     * @param imageHeight Actual image height in pixels
     * @return Array of [depth, perpendicularDistance, verticalOffset] in meters
     */
    public static double[] calculateDepthWithDimensions(
            StereoDetectedObject leftDetection,
            StereoDetectedObject rightDetection,
            StereoConfiguration config,
            int imageWidth,
            int imageHeight) {
        
        double disparity = leftDetection.centerX - rightDetection.centerX;
        
        if (disparity < MIN_DISPARITY) {
            return new double[] {Double.POSITIVE_INFINITY, 0, 0};
        }
        
        double baseline = config.getBaseline();
        double avgHorizontalFOV = (config.leftFOV.horizontalFOV + config.rightFOV.horizontalFOV) / 2.0;
        double avgVerticalFOV = (config.leftFOV.verticalFOV + config.rightFOV.verticalFOV) / 2.0;
        
        // Calculate focal length in pixels
        double focalLengthX = imageWidth / (2.0 * Math.tan(Math.toRadians(avgHorizontalFOV / 2.0)));
        double focalLengthY = imageHeight / (2.0 * Math.tan(Math.toRadians(avgVerticalFOV / 2.0)));
        
        // Calculate depth
        double depth = (baseline * focalLengthX) / disparity;
        depth = Math.min(depth, MAX_DEPTH);
        
        // Calculate perpendicular distance
        double avgCenterX = (leftDetection.centerX + rightDetection.centerX) / 2.0;
        double xOffsetPixels = avgCenterX - imageWidth / 2.0;
        double perpendicularDistance = (xOffsetPixels * depth) / focalLengthX;
        
        // Calculate vertical offset
        double avgCenterY = (leftDetection.centerY + rightDetection.centerY) / 2.0;
        double yOffsetPixels = avgCenterY - imageHeight / 2.0;
        double verticalOffset = (yOffsetPixels * depth) / focalLengthY;
        
        return new double[] {depth, perpendicularDistance, verticalOffset};
    }
}
