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

import org.opencv.core.Rect2d;

/**
 * Represents a detected object from a single camera in the stereo pair,
 * with descriptor information for matching across cameras.
 */
public class StereoDetectedObject {
    /** The bounding box of the detection in pixel coordinates */
    public final Rect2d boundingBox;
    
    /** The class index from the object detector */
    public final int classIdx;
    
    /** The class name from the object detector */
    public final String className;
    
    /** Detection confidence score (0-1) */
    public final double confidence;
    
    /** Center X coordinate in pixels */
    public final double centerX;
    
    /** Center Y coordinate in pixels */
    public final double centerY;
    
    /** Descriptor for matching: average pixel gradient magnitude around center */
    public final double avgGradient;
    
    /** Descriptor for matching: average R value in patch around center */
    public final double avgR;
    
    /** Descriptor for matching: average G value in patch around center */
    public final double avgG;
    
    /** Descriptor for matching: average B value in patch around center */
    public final double avgB;
    
    /** Which camera this detection came from (true = left, false = right) */
    public final boolean isLeftCamera;

    public StereoDetectedObject(
            Rect2d boundingBox,
            int classIdx,
            String className,
            double confidence,
            double centerX,
            double centerY,
            double avgGradient,
            double avgR,
            double avgG,
            double avgB,
            boolean isLeftCamera) {
        this.boundingBox = boundingBox;
        this.classIdx = classIdx;
        this.className = className;
        this.confidence = confidence;
        this.centerX = centerX;
        this.centerY = centerY;
        this.avgGradient = avgGradient;
        this.avgR = avgR;
        this.avgG = avgG;
        this.avgB = avgB;
        this.isLeftCamera = isLeftCamera;
    }

    /**
     * Calculate descriptor similarity between this object and another.
     * Uses Euclidean distance in descriptor space.
     * 
     * @param other The other detected object to compare
     * @return Similarity score (lower = more similar)
     */
    public double descriptorDistance(StereoDetectedObject other) {
        // Normalize gradient to 0-255 range for comparison
        double gradDiff = (this.avgGradient - other.avgGradient) / 255.0;
        double rDiff = (this.avgR - other.avgR) / 255.0;
        double gDiff = (this.avgG - other.avgG) / 255.0;
        double bDiff = (this.avgB - other.avgB) / 255.0;
        
        return Math.sqrt(gradDiff * gradDiff + rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
    }

    @Override
    public String toString() {
        return "StereoDetectedObject{"
                + "class='" + className + '\''
                + ", confidence=" + String.format("%.2f", confidence)
                + ", center=(" + String.format("%.1f", centerX) + ", " + String.format("%.1f", centerY) + ")"
                + ", camera=" + (isLeftCamera ? "left" : "right")
                + '}';
    }
}
