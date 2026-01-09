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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Configuration for a stereo camera pair. Contains information about camera names,
 * positions, rotations, and field of view settings.
 */
public class StereoConfiguration {
    /** Unique name of the left camera */
    public final String leftCameraName;
    
    /** Unique name of the right camera */
    public final String rightCameraName;
    
    /** Position of left camera relative to center point (x, y, z in meters) */
    public final CameraPosition leftPosition;
    
    /** Position of right camera relative to center point (x, y, z in meters) */
    public final CameraPosition rightPosition;
    
    /** Field of view settings for left camera */
    public final CameraFOV leftFOV;
    
    /** Field of view settings for right camera */
    public final CameraFOV rightFOV;

    @JsonCreator
    public StereoConfiguration(
            @JsonProperty("leftCameraName") String leftCameraName,
            @JsonProperty("rightCameraName") String rightCameraName,
            @JsonProperty("leftPosition") CameraPosition leftPosition,
            @JsonProperty("rightPosition") CameraPosition rightPosition,
            @JsonProperty("leftFOV") CameraFOV leftFOV,
            @JsonProperty("rightFOV") CameraFOV rightFOV) {
        this.leftCameraName = leftCameraName;
        this.rightCameraName = rightCameraName;
        this.leftPosition = leftPosition;
        this.rightPosition = rightPosition;
        this.leftFOV = leftFOV;
        this.rightFOV = rightFOV;
    }

    /**
     * Calculate the baseline distance between the two cameras.
     * @return The distance between camera centers in meters
     */
    public double getBaseline() {
        double dx = rightPosition.x - leftPosition.x;
        double dy = rightPosition.y - leftPosition.y;
        double dz = rightPosition.z - leftPosition.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Represents the position and rotation of a camera relative to the stereo pair center.
     */
    public static class CameraPosition {
        /** X position in meters (positive = right) */
        public final double x;
        /** Y position in meters (positive = down) */
        public final double y;
        /** Z position in meters (positive = forward) */
        public final double z;
        /** Rotation around X axis in degrees (pitch) */
        public final double rotX;
        /** Rotation around Y axis in degrees (yaw) */
        public final double rotY;
        /** Rotation around Z axis in degrees (roll) */
        public final double rotZ;

        @JsonCreator
        public CameraPosition(
                @JsonProperty("x") double x,
                @JsonProperty("y") double y,
                @JsonProperty("z") double z,
                @JsonProperty("rotX") double rotX,
                @JsonProperty("rotY") double rotY,
                @JsonProperty("rotZ") double rotZ) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.rotX = rotX;
            this.rotY = rotY;
            this.rotZ = rotZ;
        }
    }

    /**
     * Represents the field of view settings for a camera.
     */
    public static class CameraFOV {
        /** Horizontal field of view in degrees */
        public final double horizontalFOV;
        /** Vertical field of view in degrees */
        public final double verticalFOV;

        @JsonCreator
        public CameraFOV(
                @JsonProperty("horizontalFOV") double horizontalFOV,
                @JsonProperty("verticalFOV") double verticalFOV) {
            this.horizontalFOV = horizontalFOV;
            this.verticalFOV = verticalFOV;
        }
    }

    @Override
    public String toString() {
        return "StereoConfiguration{"
                + "leftCamera='" + leftCameraName + '\''
                + ", rightCamera='" + rightCameraName + '\''
                + ", baseline=" + getBaseline()
                + "m}";
    }
}
