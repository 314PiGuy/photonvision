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

/**
 * Represents a matched pair of detected objects from the left and right cameras.
 * Contains computed depth and position information.
 */
public class StereoMatchedPair {
    /** The detection from the left camera */
    public final StereoDetectedObject leftDetection;
    
    /** The detection from the right camera */
    public final StereoDetectedObject rightDetection;
    
    /** Match ID for display purposes */
    public final int matchId;
    
    /** Computed depth (distance along Z axis) in meters */
    public final double depth;
    
    /** Perpendicular distance from center line in meters (positive = right) */
    public final double perpendicularDistance;
    
    /** Vertical offset from center in meters (positive = down) */
    public final double verticalOffset;
    
    /** Combined confidence score (average of both detections) */
    public final double averageConfidence;
    
    /** Descriptor match quality (lower = better match) */
    public final double matchQuality;

    public StereoMatchedPair(
            StereoDetectedObject leftDetection,
            StereoDetectedObject rightDetection,
            int matchId,
            double depth,
            double perpendicularDistance,
            double verticalOffset,
            double matchQuality) {
        this.leftDetection = leftDetection;
        this.rightDetection = rightDetection;
        this.matchId = matchId;
        this.depth = depth;
        this.perpendicularDistance = perpendicularDistance;
        this.verticalOffset = verticalOffset;
        this.averageConfidence = (leftDetection.confidence + rightDetection.confidence) / 2.0;
        this.matchQuality = matchQuality;
    }

    /**
     * Get the class name of the matched detection.
     * @return The class name
     */
    public String getClassName() {
        return leftDetection.className;
    }

    /**
     * Get the class index of the matched detection.
     * @return The class index
     */
    public int getClassIdx() {
        return leftDetection.classIdx;
    }

    @Override
    public String toString() {
        return "StereoMatchedPair{"
                + "id=" + matchId
                + ", class='" + getClassName() + '\''
                + ", confidence=" + String.format("%.2f", averageConfidence)
                + ", depth=" + String.format("%.2f", depth) + "m"
                + ", perpDist=" + String.format("%.2f", perpendicularDistance) + "m"
                + ", matchQuality=" + String.format("%.3f", matchQuality)
                + '}';
    }
}
