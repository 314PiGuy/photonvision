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

package org.photonvision.vision.frame.provider;

import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.cscore.CvSink;
import edu.wpi.first.cscore.UsbCamera;
import edu.wpi.first.networktables.BooleanSubscriber;
import edu.wpi.first.util.PixelFormat;
import edu.wpi.first.util.RawFrame;
import org.opencv.core.Mat;
import org.photonvision.common.dataflow.networktables.NetworkTablesManager;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.CscoreExtras;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.processes.VisionSourceSettables;

public class USBFrameProvider extends CpuImageProcessor {
    private final Logger logger;

    private UsbCamera camera = null;
    private CvSink cvSink = null;

    @SuppressWarnings("SpellCheckingInspection")
    private VisionSourceSettables settables;

    private Runnable connectedCallback;

    private long lastTime = 0;

    // CSCore capture timestamps can come from a different timebase than MathUtils.wpiNanoTime().
    // We estimate a translation offset at runtime so downstream latency calculations are sane.
    private volatile boolean captureTimeOffsetInitialized = false;
    private volatile long captureTimeOffsetNs = 0;
    private long lastOffsetLogTime = 0;

    private long normalizeCaptureTimestampNs(long captureTimeNs) {
        if (captureTimeNs <= 0) return captureTimeNs;

        long nowNs = MathUtils.wpiNanoTime();
        long offset = nowNs - captureTimeNs;

        // Initialize or low-pass filter to avoid jitter.
        if (!captureTimeOffsetInitialized) {
            captureTimeOffsetInitialized = true;
            captureTimeOffsetNs = offset;
        } else {
            // If the offset jumps massively (e.g. source restarted), reset.
            long prev = captureTimeOffsetNs;
            if (Math.abs(offset - prev) > 30_000_000_000L) {
                captureTimeOffsetNs = offset;
            } else {
                // 90/10 smoothing
                captureTimeOffsetNs = (prev * 9L + offset) / 10L;
            }
        }

        // Log offset periodically for diagnostics
        long nowMs = System.currentTimeMillis();
        if (nowMs - lastOffsetLogTime > 5000) {
            lastOffsetLogTime = nowMs;
            logger.info(String.format("Capture timestamp offset: %.3fs (raw=%dns, normalized=%dns, now=%dns)",
                captureTimeOffsetNs / 1_000_000_000.0,
                captureTimeNs,
                captureTimeNs + captureTimeOffsetNs,
                MathUtils.wpiNanoTime()));
        }

        return captureTimeNs + captureTimeOffsetNs;
    }

    // subscribers are lightweight, and I'm lazy
    private final BooleanSubscriber useNewBehaviorSub;

    @SuppressWarnings("SpellCheckingInspection")
    public USBFrameProvider(
            UsbCamera camera, VisionSourceSettables visionSettables, Runnable connectedCallback) {
        this.camera = camera;
        this.cvSink = CameraServer.getVideo(this.camera);
        this.logger =
                new Logger(
                        USBFrameProvider.class, visionSettables.getConfiguration().nickname, LogGroup.Camera);
        this.cvSink.setEnabled(true);

        this.settables = visionSettables;

        var useNewBehaviorTopic =
                NetworkTablesManager.getInstance().kRootTable.getBooleanTopic("use_new_cscore_frametime");

        useNewBehaviorSub = useNewBehaviorTopic.subscribe(false);
        this.connectedCallback = connectedCallback;
    }

    @Override
    public boolean checkCameraConnected() {
        boolean connected = camera.isConnected();

        if (!cameraPropertiesCached && connected) {
            logger.info("Camera connected! running callback");
            onCameraConnected();
        }

        return connected;
    }

    final double CSCORE_DEFAULT_FRAME_TIMEOUT = 1.0 / 4.0;

    @Override
    public CapturedFrame getInputMat() {
        if (!cameraPropertiesCached && camera.isConnected()) {
            onCameraConnected();
        }

        if (!useNewBehaviorSub.get()) {
            // We allocate memory so we don't fill a Mat in use by another thread (memory model is easier)
            var mat = new CVMat();
            // This is from wpi::Now, or WPIUtilJNI.now(). The epoch from grabFrame is uS since
            // Hal::initialize was called
            // TODO - under the hood, this incurs an extra copy. We should avoid this, if we
            // can.
            long captureTimeNs = cvSink.grabFrame(mat.getMat(), CSCORE_DEFAULT_FRAME_TIMEOUT) * 1000;
            captureTimeNs = normalizeCaptureTimestampNs(captureTimeNs);

            if (captureTimeNs == 0) {
                var error = cvSink.getError();
                logger.error("Error grabbing image: " + error);
            }

            return new CapturedFrame(mat, settables.getFrameStaticProperties(), captureTimeNs);
        } else {
            // We allocate memory so we don't fill a Mat in use by another thread (memory model is easier)
            // TODO - consider a frame pool
            // TODO - getCurrentVideoMode is a JNI call for us, but profiling indicates it's fast
            var cameraMode = settables.getCurrentVideoMode();
            var frame = new RawFrame();
            frame.setInfo(
                    cameraMode.width,
                    cameraMode.height,
                    // hard-coded 3 channel
                    cameraMode.width * 3,
                    PixelFormat.kBGR);

            // This is from wpi::Now, or WPIUtilJNI.now(). The epoch from grabFrame is uS since
            // Hal::initialize was called
            long captureTimeUs =
                    CscoreExtras.grabRawSinkFrameTimeoutLastTime(
                            cvSink.getHandle(), frame.getNativeObj(), CSCORE_DEFAULT_FRAME_TIMEOUT, lastTime);
            lastTime = captureTimeUs;

            CVMat ret;

            if (captureTimeUs == 0) {
                var error = cvSink.getError();
                logger.error("Error grabbing image: " + error);

                frame.close();
                ret = new CVMat();
            } else {
                // No error! yay
                var mat = new Mat(CscoreExtras.wrapRawFrame(frame.getNativeObj()));

                ret = new CVMat(mat, frame);
            }

            long captureTimeNs = normalizeCaptureTimestampNs(captureTimeUs * 1000);

            return new CapturedFrame(ret, settables.getFrameStaticProperties(), captureTimeNs);
        }
    }

    @Override
    public String getName() {
        return "USBFrameProvider - " + cvSink.getName();
    }

    @Override
    public void release() {
        CameraServer.removeServer(cvSink.getName());
        cvSink.close();
        cvSink = null;
    }

    @Override
    public void onCameraConnected() {
        super.onCameraConnected();

        this.connectedCallback.run();
    }

    @Override
    public boolean isConnected() {
        return camera.isConnected();
    }

    public void updateSettables(VisionSourceSettables settables) {
        this.settables = settables;
    }
}
