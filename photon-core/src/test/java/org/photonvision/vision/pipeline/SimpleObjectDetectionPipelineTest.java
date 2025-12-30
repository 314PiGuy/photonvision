package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;
import org.photonvision.vision.frame.Frame;

public class SimpleObjectDetectionPipelineTest {

    @Test
    public void testSimplePipelineRuns() {
        SimpleObjectDetectionPipeline p = new SimpleObjectDetectionPipeline();
        // use default settings (null) which should fall back to NullModel safely
        Frame f = new Frame();
        assertDoesNotThrow(() -> {
            var res = p.run(f, null);
            assertNotNull(res);
        });
    }
}
