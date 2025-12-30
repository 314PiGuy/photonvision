package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.CVMat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Simulate multiple camera streams feeding blank Mats into independent pipelines concurrently.
 */
public class MultiplePipelinesStreamTest {

    @org.junit.jupiter.api.BeforeAll
    public static void ensureNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(org.photonvision.common.util.TestUtils.loadLibraries(), "Native libraries not available");
    }

    @Test
    public void testMultiplePipelinesRunConcurrently() throws Exception {
        int pipelineCount = 6;
        int framesPerPipeline = 30;

        List<SimpleObjectDetectionPipeline> pipelines = new ArrayList<>();
        for (int i = 0; i < pipelineCount; i++) pipelines.add(new SimpleObjectDetectionPipeline());

        int before = CVMat.getMatCount();

        ExecutorService ex = Executors.newFixedThreadPool(pipelineCount);
        List<Future<?>> futures = new ArrayList<>();
        for (SimpleObjectDetectionPipeline p : pipelines) {
            // ensure settings are present for the simple pipeline
            p.setSettings(new CVPipelineSettings());
            futures.add(
                    ex.submit(
                            () -> {
                                for (int j = 0; j < framesPerPipeline; j++) {
                                    Frame f = new Frame();
                                    f.colorImage.getMat().create(120, 160, CvType.CV_8UC3);
                                    assertDoesNotThrow(() -> {
                                        var res = p.run(f, null);
                                        res.release();
                                    });
                                    f.release();
                                }
                            }));
        }

        for (Future<?> future : futures) future.get(30, TimeUnit.SECONDS);

        ex.shutdown();
        ex.awaitTermination(10, TimeUnit.SECONDS);

        int after = CVMat.getMatCount();
        int delta = after - before;
        System.out.println("MultiplePipelinesStreamTest: mats before=" + before + ", after=" + after + ", delta=" + delta);

        // Sanity check: total mats created should be proportional to frames processed
        int framesTotal = pipelineCount * framesPerPipeline;
        int allowed = framesTotal * 3 + pipelineCount * 2; // allow up to 3 mats per frame plus some overhead
        assertTrue(delta <= allowed, "Unexpectedly high mat allocation during stream (delta=" + delta + ", allowed=" + allowed + ")");

        // Release pipelines as a best-effort cleanup
        for (SimpleObjectDetectionPipeline p : pipelines) p.release();
        System.gc();
        Thread.sleep(200);
    }
}
