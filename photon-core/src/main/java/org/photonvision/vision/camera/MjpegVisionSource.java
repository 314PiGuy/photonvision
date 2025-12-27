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

package org.photonvision.vision.camera;

import edu.wpi.first.cscore.CvSink;
import edu.wpi.first.cscore.HttpCamera;
import edu.wpi.first.cscore.VideoMode;
import edu.wpi.first.util.PixelFormat;
import java.util.HashMap;
import org.opencv.core.Mat;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.camera.PVCameraInfo.PVMjpegCameraInfo;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.processes.VisionSource;
import org.photonvision.vision.processes.VisionSourceSettables;

public class MjpegVisionSource extends VisionSource {
    private static final Logger logger = new Logger(MjpegVisionSource.class, LogGroup.Camera);
    private final PVMjpegCameraInfo info;
    private final MjpegFrameProvider frameProvider;

    public MjpegVisionSource(CameraConfiguration config) {
        super(config);
        if (!(config.matchedCameraInfo instanceof PVMjpegCameraInfo)) {
            throw new IllegalArgumentException("Config must be for an MJPEG camera");
        }
        this.info = (PVMjpegCameraInfo) config.matchedCameraInfo;
        this.frameProvider = new MjpegFrameProvider(config);
    }

    @Override
    public FrameProvider getFrameProvider() {
        return frameProvider;
    }

    @Override
    public VisionSourceSettables getSettables() {
        return new VisionSourceSettables(cameraConfiguration) {
            @Override
            public HashMap<Integer, VideoMode> getAllVideoModes() {
                HashMap<Integer, VideoMode> modes = new HashMap<>();
                modes.put(0, new VideoMode(PixelFormat.kMJPEG, 640, 480, 30));
                return modes;
            }

            @Override
            public void setExposureRaw(double exposureRaw) {}

            @Override
            public double getMinExposureRaw() { return 0; }

            @Override
            public double getMaxExposureRaw() { return 100; }

            @Override
            public void setAutoExposure(boolean cameraAutoExposure) {}

            @Override
            public void setWhiteBalanceTemp(double temp) {}

            @Override
            public void setAutoWhiteBalance(boolean autowb) {}

            @Override
            public void setBrightness(int brightness) {}

            @Override
            public void setGain(int gain) {}

            @Override
            public VideoMode getCurrentVideoMode() {
                return new VideoMode(PixelFormat.kMJPEG, 640, 480, 30);
            }

            @Override
            public void setVideoModeInternal(VideoMode videoMode) {}

            @Override
            public double getMinWhiteBalanceTemp() { return 0; }

            @Override
            public double getMaxWhiteBalanceTemp() { return 10000; }
        };
    }

    @Override
    public void remakeSettables() {
        // No-op
    }

    @Override
    public boolean hasLEDs() {
        return false;
    }

    @Override
    public boolean isVendorCamera() {
        return false;
    }

    @Override
    public void release() {
        frameProvider.release();
    }

    private static class MjpegFrameProvider extends FrameProvider {
        private final PVMjpegCameraInfo info;
        private final CameraConfiguration config;
        private HttpCamera camera;
        private CvSink cvSink;
        private final Mat tempMat = new Mat();
        private boolean connected = false;
        private FrameStaticProperties frameProps;

        public MjpegFrameProvider(CameraConfiguration config) {
            this.config = config;
            this.info = (PVMjpegCameraInfo) config.matchedCameraInfo;
        }

        @Override
        public Frame get() {
            if (!connected) {
                if (!checkCameraConnected()) {
                    return new Frame(0, null, null, FrameThresholdType.NONE, 0, null);
                }
            }

            long result = cvSink.grabFrame(tempMat, 1.0);
            if (result != 0) {
                if (!tempMat.empty()) {
                    if (frameProps == null || frameProps.imageWidth != tempMat.width() || frameProps.imageHeight != tempMat.height()) {
                        var cal = config.calibrations.isEmpty() ? null : config.calibrations.get(0);
                        frameProps = new FrameStaticProperties(new VideoMode(PixelFormat.kMJPEG, tempMat.width(), tempMat.height(), 30), config.FOV, cal);
                    }

                    CVMat cvMat = new CVMat(tempMat.clone());
                    return new Frame(0, cvMat, null, FrameThresholdType.NONE, System.nanoTime(), frameProps);
                }
            } else {
                // Error grabbing frame
                String error = cvSink.getError();
                logger.error("Error grabbing frame from MJPEG stream: " + error);
                connected = false;
                if (camera != null) {
                    camera.close();
                    camera = null;
                }
                if (cvSink != null) {
                    cvSink.close();
                    cvSink = null;
                }
            }
            return new Frame(0, null, null, FrameThresholdType.NONE, 0, null);
        }

        @Override
        public boolean isConnected() {
            return connected;
        }

        @Override
        public boolean checkCameraConnected() {
            if (connected) return true;
            try {
                logger.info("Attempting to connect to MJPEG stream: " + info.url);
                
                if (camera == null) {
                    camera = new HttpCamera(info.name, info.url);
                }
                if (cvSink == null) {
                    cvSink = new CvSink("MjpegSink");
                    cvSink.setSource(camera);
                    cvSink.setEnabled(true);
                }

                if (camera.isValid()) {
                    connected = true;
                    logger.info("Connected to MJPEG stream!");
                    onCameraConnected();
                    return true;
                } else {
                    logger.error("Failed to open MJPEG stream (isValid=false): " + info.url);
                }
            } catch (Exception e) {
                logger.error("Failed to connect to MJPEG stream", e);
            }
            return false;
        }

        @Override
        public String getName() {
            return info.name;
        }

        @Override
        public void requestFrameThresholdType(FrameThresholdType type) {}

        @Override
        public void requestFrameRotation(ImageRotationMode rotationMode) {}

        @Override
        public void requestFrameCopies(boolean copyInput, boolean copyOutput) {}

        @Override
        public void requestHsvSettings(HSVPipe.HSVParams params) {}

        @Override
        public void release() {
            if (camera != null) {
                camera.close();
            }
            if (cvSink != null) {
                cvSink.close();
            }
            tempMat.release();
        }
    }
}
