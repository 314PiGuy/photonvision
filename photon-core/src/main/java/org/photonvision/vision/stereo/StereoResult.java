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

import java.util.List;

/**
 * Result from stereo vision processing.
 * Contains matched pairs with depth information and processing metadata.
 */
public class StereoResult {
    /** Timestamp of the result */
    public final long timestampNanos;
    
    /** List of matched object pairs with depth information */
    public final List<StereoMatchedPair> matchedPairs;
    
    /** Number of detections from left camera */
    public final int leftDetectionCount;
    
    /** Number of detections from right camera */
    public final int rightDetectionCount;
    
    /** Processing time in nanoseconds */
    public final long processingTimeNanos;
    
    /** Frames per second estimate */
    public final double fps;
    
    /** Whether the result is valid (both cameras had frames) */
    public final boolean isValid;
    
    /** Left camera image width */
    public final int leftImageWidth;
    
    /** Left camera image height */
    public final int leftImageHeight;
    
    /** Right camera image width */
    public final int rightImageWidth;
    
    /** Right camera image height */
    public final int rightImageHeight;

    public StereoResult(
            long timestampNanos,
            List<StereoMatchedPair> matchedPairs,
            int leftDetectionCount,
            int rightDetectionCount,
            long processingTimeNanos,
            double fps,
            boolean isValid,
            int leftImageWidth,
            int leftImageHeight,
            int rightImageWidth,
            int rightImageHeight) {
        this.timestampNanos = timestampNanos;
        this.matchedPairs = matchedPairs;
        this.leftDetectionCount = leftDetectionCount;
        this.rightDetectionCount = rightDetectionCount;
        this.processingTimeNanos = processingTimeNanos;
        this.fps = fps;
        this.isValid = isValid;
        this.leftImageWidth = leftImageWidth;
        this.leftImageHeight = leftImageHeight;
        this.rightImageWidth = rightImageWidth;
        this.rightImageHeight = rightImageHeight;
    }

    /**
     * Create an invalid/empty result.
     */
    public static StereoResult empty() {
        return new StereoResult(
                System.nanoTime(),
                List.of(),
                0, 0,
                0, 0,
                false,
                0, 0, 0, 0);
    }

    /**
     * Check if there are any matched pairs.
     */
    public boolean hasMatches() {
        return matchedPairs != null && !matchedPairs.isEmpty();
    }

    /**
     * Get the number of matched pairs.
     */
    public int getMatchCount() {
        return matchedPairs != null ? matchedPairs.size() : 0;
    }

    @Override
    public String toString() {
        return "StereoResult{"
                + "valid=" + isValid
                + ", matches=" + getMatchCount()
                + ", leftDet=" + leftDetectionCount
                + ", rightDet=" + rightDetectionCount
                + ", fps=" + String.format("%.1f", fps)
                + ", procTime=" + String.format("%.2f", processingTimeNanos / 1_000_000.0) + "ms"
                + '}';
    }
}
