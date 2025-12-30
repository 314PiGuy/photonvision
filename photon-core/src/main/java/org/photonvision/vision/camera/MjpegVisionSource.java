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

import edu.wpi.first.cscore.VideoMode;
import edu.wpi.first.util.PixelFormat;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
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
            throw new IllegalArgumentException("Config must be for an MJPEG cameraConnection");
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
            public double getMinExposureRaw() {
                return 0;
            }

            @Override
            public double getMaxExposureRaw() {
                return 100;
            }

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
            public double getMinWhiteBalanceTemp() {
                return 0;
            }

            @Override
            public double getMaxWhiteBalanceTemp() {
                return 10000;
            }
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
        private final org.photonvision.vision.camera.PVCameraInfo.PVMjpegCameraInfo info;
        private final CameraConfiguration config;
        private HttpURLConnection cameraConnection = null;
        private boolean connected = false;
        private FrameStaticProperties frameProps;

        private InputStream in;
        private ByteArrayOutputStream img = new ByteArrayOutputStream();

        private int prev = 0, cur;
        private boolean capture = false;
        private long frameCount = 0;

        public MjpegFrameProvider(CameraConfiguration config) {
            this.config = config;
            this.info = (PVMjpegCameraInfo) config.matchedCameraInfo;
        }

        private Mat readMat() throws IOException {
            while ((cur = in.read()) != -1) {
                if (prev == 0xFF && cur == 0xD8) { // JPEG SOI
                    img.reset();
                    img.write(prev);
                    capture = true;
                }
                if (capture) img.write(cur);

                if (prev == 0xFF && cur == 0xD9 && capture) { // JPEG EOI
                    byte[] jpeg = img.toByteArray();
                    MatOfByte mob = new MatOfByte(jpeg);
                    Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
                    mob.release();
                    capture = false;
                    prev = cur;
                    return mat;
                }

                prev = cur;
            }
            return null;
        }

        @Override
        public Frame get() {
            if (!connected) {
                if (!checkCameraConnected()) {
                    return new Frame(0, null, null, FrameThresholdType.NONE, 0, null);
                }
            }

            try {
                Mat mat = readMat();
                if (mat == null) {
                    throw new IOException("Stream closed");
                }
                if (mat.empty()) {
                    mat.release();
                    return new Frame(0, null, null, FrameThresholdType.NONE, 0, null);
                }

                if (frameProps == null
                        || frameProps.imageWidth != mat.width()
                        || frameProps.imageHeight != mat.height()) {
                    var cal = config.calibrations.isEmpty() ? null : config.calibrations.get(0);
                    frameProps =
                            new FrameStaticProperties(
                                    new VideoMode(PixelFormat.kMJPEG, mat.width(), mat.height(), 30),
                                    config.FOV,
                                    cal);
                }
                System.out.println("Got MJPEG frame: " + mat.size());
                return new Frame(
                        frameCount++,
                        new CVMat(mat),
                        new CVMat(mat.clone()),
                        FrameThresholdType.NONE,
                        System.nanoTime(),
                        frameProps);
            } catch (Exception e) {
                // Error grabbing frame
                logger.error("Error grabbing frame from MJPEG stream", e);
                connected = false;
                if (cameraConnection != null) {
                    cameraConnection.disconnect();
                    cameraConnection = null;
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
                if (cameraConnection == null) {
                    HttpURLConnection conn = (HttpURLConnection) new URL(info.url).openConnection();
                    conn.setRequestProperty("User-Agent", "Java");
                    conn.connect();
                    cameraConnection = conn;
                    in = new BufferedInputStream(cameraConnection.getInputStream());
                    System.out.println("Connected to MJPEG stream");
                    connected = true;
                    onCameraConnected();
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
            if (cameraConnection != null) {
                cameraConnection.disconnect();
            }
            connected = false;
        }
    }
}
