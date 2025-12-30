package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.LinkedList;
import org.junit.jupiter.api.Test;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.configuration.NeuralNetworkPropertyManager;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

public class ObjectDetectionPipelineTest {

    @Test
    public void testRunWithNullModelPathDoesNotThrow() {
        ObjectDetectionPipeline pipeline = new ObjectDetectionPipeline();
        ObjectDetectionPipelineSettings settings = pipeline.getSettings();

        // Simulate a legacy ModelProperties with a null modelPath
        var props =
                new NeuralNetworkPropertyManager.ModelProperties(
                        null,
                        "legacy",
                        new LinkedList<>(),
                        640,
                        480,
                        NeuralNetworkModelManager.Family.ONNX,
                        NeuralNetworkModelManager.Version.YOLOV8);
        settings.model = props;

        Frame frame = new Frame();

        assertDoesNotThrow(
                () -> {
                    CVPipelineResult res = pipeline.run(frame, null);
                    assertNotNull(res);
                });
    }
}
