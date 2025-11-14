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

package org.photonvision.onnx;

import org.opencv.core.Point;
import org.opencv.core.Rect2d;

/** JNI bindings for PhotonVision's ONNX object detector. */
public class OnnxJNI {
    /** Represents a single detection result returned by the ONNX backend. */
    public static class OnnxResult {
        public final Rect2d rect;
        public final float confidence;
        public final int classId;

        public OnnxResult(int x1, int y1, int x2, int y2, float confidence, int classId) {
            this.confidence = confidence;
            this.classId = classId;
            this.rect = new Rect2d(new Point(x1, y1), new Point(x2, y2));
        }

        @Override
        public String toString() {
            return "OnnxResult{"
                    + "rect="
                    + rect
                    + ", confidence="
                    + confidence
                    + ", classId="
                    + classId
                    + '}';
        }
    }

    public static native long create(String modelPath);

    public static native void destroy(long ptr);

    public static native OnnxResult[] detect(
            long detectorPtr, long imagePtr, double boxThresh, double nmsThreshold, int classCount);

    public static native boolean isQuantized(long detectorPtr);

    public static native double getLastMaxScore(long detectorPtr);

    public static native int[] getInputSize(long detectorPtr);
}
