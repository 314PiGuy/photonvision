package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe;

/**
 * Tests that simulate a camera stream by repeatedly feeding blank Mats into pipelines and asserting
 * that detectors are not recreated repeatedly and that resource usage stays bounded.
 */
public class ObjectDetectionPipelineStreamTest {

    @org.junit.jupiter.api.BeforeAll
    public static void ensureNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                org.photonvision.common.util.TestUtils.loadLibraries(), "Native libraries not available");
    }

    @Test
    public void testObjectDetectionPipelineProcessesBlankFrames() throws Exception {
        ObjectDetectionPipeline pipeline = new ObjectDetectionPipeline();
        // Ensure settings has no model (legacy/empty config)
        pipeline.getSettings().model = null;

        int beforeMats = CVMat.getMatCount();

        // Ensure the internal detector is a NullModel initially
        Field ocPipeField = ObjectDetectionPipeline.class.getDeclaredField("objectDetectorPipe");
        ocPipeField.setAccessible(true);
        ObjectDetectionPipe ocPipe = (ObjectDetectionPipe) ocPipeField.get(pipeline);

        Field detectorField = ObjectDetectionPipe.class.getDeclaredField("detector");
        detectorField.setAccessible(true);
        Object detectorBefore = detectorField.get(ocPipe);
        assertTrue(detectorBefore instanceof NullModel, "Detector should start as NullModel");

        // Simulate a short stream of frames (blank / black)
        for (int i = 0; i < 50; i++) {
            Frame f = new Frame();
            f.colorImage.getMat().create(360, 640, CvType.CV_8UC3); // non-empty mat

            // Should not throw
            assertDoesNotThrow(
                    () -> {
                        var res = pipeline.run(f, null);
                        // release result to avoid leaking Mats
                        res.release();
                    });

            // Release test-owned frame resources
            f.release();
        }

        // After running the stream, ensure mat allocation count did not grow unboundedly
        int afterMats = CVMat.getMatCount();
        int delta = afterMats - beforeMats;
        System.out.println(
                "ObjectDetectionPipelineStreamTest: mats before="
                        + beforeMats
                        + ", after="
                        + afterMats
                        + ", delta="
                        + delta);

        // Sanity check: expect at most ~3 mats per frame plus small overhead
        int frames = 50;
        int allowed = frames * 3 + 5;
        assertTrue(
                delta <= allowed,
                "Unexpected CVMat allocation during stream (delta=" + delta + ", allowed=" + allowed + ")");

        // Release pipeline as a best-effort cleanup
        pipeline.release();
        System.gc();
        Thread.sleep(200);

        int afterCleanup = CVMat.getMatCount();
        System.out.println("ObjectDetectionPipelineStreamTest: after cleanup mats=" + afterCleanup);

        // Ensure detector is still NullModel (wasn't eagerly replaced by a heavy detector)
        Object detectorAfter = detectorField.get(ocPipe);
        assertEquals(
                detectorBefore,
                detectorAfter,
                "Detector instance should not have changed when model is missing");
    }

    @Test
    public void testSimpleObjectDetectionPipelineStreamRuns() {
        SimpleObjectDetectionPipeline p = new SimpleObjectDetectionPipeline();
        p.setSettings(new CVPipelineSettings());

        int beforeMats = CVMat.getMatCount();

        for (int i = 0; i < 25; i++) {
            Frame f = new Frame();
            f.colorImage.getMat().create(240, 320, CvType.CV_8UC3);
            assertDoesNotThrow(
                    () -> {
                        var res = p.run(f, null);
                        res.release();
                    });
            f.release();
        }

        int afterMats = CVMat.getMatCount();
        int delta = afterMats - beforeMats;
        System.out.println(
                "SimpleObjectDetectionPipelineTest: mats before="
                        + beforeMats
                        + ", after="
                        + afterMats
                        + ", delta="
                        + delta);
        int frames = 25;
        int allowed = frames * 3 + 5;
        assertTrue(
                delta <= allowed,
                "Unexpected CVMat allocation during simple pipeline stream (delta="
                        + delta
                        + ", allowed="
                        + allowed
                        + ")");
    }
}
