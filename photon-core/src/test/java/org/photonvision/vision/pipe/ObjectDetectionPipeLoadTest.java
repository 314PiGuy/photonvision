package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe.ObjectDetectionPipeParams;

public class ObjectDetectionPipeLoadTest {
    static class CountingModel implements Model {
        private final AtomicInteger loads = new AtomicInteger();
        private final String uid;

        public CountingModel(String uid) {
            this.uid = uid;
        }

        @Override
        public org.photonvision.vision.objects.ObjectDetector load() {
            loads.incrementAndGet();
            return NullModel.getInstance();
        }

        public int getLoadCount() {
            return loads.get();
        }

        @Override
        public String getUID() {
            return uid;
        }

        @Override
        public String getNickname() {
            return uid;
        }

        @Override
        public org.photonvision.common.configuration.NeuralNetworkModelManager.Family getFamily() {
            return null;
        }

        @Override
        public org.photonvision.common.configuration.NeuralNetworkPropertyManager.ModelProperties
                getProperties() {
            return null;
        }
    }

    @BeforeAll
    public static void ensureNativeOrSkip() {
        // We only need native libraries for CVMat allocation; skip tests otherwise
        org.junit.jupiter.api.Assumptions.assumeTrue(
                TestUtils.loadLibraries(), "Native libraries not available");
    }

    @Test
    public void testModelLoadOnlyOnce() {
        ObjectDetectionPipe pipe = new ObjectDetectionPipe();
        CountingModel cm = new CountingModel("one");

        pipe.setParams(new ObjectDetectionPipeParams(0.9, 0.45, cm));

        // run multiple times, should only call load() once
        var mat = new org.photonvision.vision.opencv.CVMat();
        for (int i = 0; i < 10; i++) {
            pipe.run(mat);
        }

        assertEquals(
                1, cm.getLoadCount(), "Model.load should be called only once for identical model uid");

        // now switch to a different model uid
        CountingModel cm2 = new CountingModel("two");
        pipe.setParams(new ObjectDetectionPipeParams(0.9, 0.45, cm2));
        pipe.run(mat);
        assertEquals(1, cm2.getLoadCount(), "Second model load should be called once after switch");
    }
}
