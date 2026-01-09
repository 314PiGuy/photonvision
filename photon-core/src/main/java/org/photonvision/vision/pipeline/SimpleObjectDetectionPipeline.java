package org.photonvision.vision.pipeline;

import java.util.List;
import java.util.Optional;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.NullModel;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe;
import org.photonvision.vision.pipe.impl.ObjectDetectionPipe.ObjectDetectionPipeParams;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/** Minimal pipeline that only runs object detection. */
public class SimpleObjectDetectionPipeline
        extends CVPipeline<CVPipelineResult, CVPipelineSettings> {
    private final ObjectDetectionPipe objectDetectionPipe = new ObjectDetectionPipe();
    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;

    private Model lastSelectedModel = NullModel.getInstance();
    private double lastConfidence = -1.0;
    private double lastNms = -1.0;

    public SimpleObjectDetectionPipeline() {
        super(PROCESSING_TYPE);
    }

    @Override
    protected void setPipeParamsImpl() {
        // For the simple pipeline we read generic settings from CVPipelineSettings if available
        double confidence = 0.9;
        double nms = 0.45;
        Model modelToUse = NullModel.getInstance();

        // If the settings object has known fields we'll try to extract them
        if (settings != null) {
            try {
                Object confidenceObj = settings.getClass().getField("confidence").get(settings);
                if (confidenceObj instanceof Number) confidence = ((Number) confidenceObj).doubleValue();
            } catch (NoSuchFieldException | IllegalAccessException ignored) {
            }
            try {
                Object nmsObj = settings.getClass().getField("nms").get(settings);
                if (nmsObj instanceof Number) nms = ((Number) nmsObj).doubleValue();
            } catch (NoSuchFieldException | IllegalAccessException ignored) {
            }
            try {
                Object modelProps = settings.getClass().getField("model").get(settings);
                if (modelProps != null) {
                    // Attempt to get a model path string
                    try {
                        var mpField = modelProps.getClass().getMethod("modelPath");
                        Object mpVal = mpField.invoke(modelProps);
                        if (mpVal != null) {
                            Optional<Model> maybe =
                                    NeuralNetworkModelManager.getInstance().getModel(mpVal.toString());
                            modelToUse = maybe.orElse(NullModel.getInstance());
                        }
                    } catch (Exception ignored) {
                    }
                }
            } catch (NoSuchFieldException | IllegalAccessException ignored) {
            }
        }

        if (modelToUse != lastSelectedModel || confidence != lastConfidence || nms != lastNms) {
            objectDetectionPipe.setParams(new ObjectDetectionPipeParams(confidence, nms, modelToUse));
            lastSelectedModel = modelToUse;
            lastConfidence = confidence;
            lastNms = nms;
        }
    }

    @Override
    protected CVPipelineResult process(Frame frame, CVPipelineSettings settings) {
        long sum = 0;
        var nnResult = objectDetectionPipe.run(frame.colorImage);
        sum += nnResult.nanosElapsed;

        // For simplicity this minimal pipeline does not convert detections to TrackedTargets.
        List<String> classNames = objectDetectionPipe.getClassNames();
        return new CVPipelineResult(frame.sequenceID, sum, 0.0, List.of(), frame, classNames);
    }

    @Override
    public void release() {
        objectDetectionPipe.release();
        super.release();
    }
}
