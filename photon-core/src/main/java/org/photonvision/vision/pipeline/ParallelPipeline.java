package org.photonvision.vision.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;

/**
 * Executes a list of child pipelines in parallel (conceptually). Each child receives a cloned
 * frame. The output frame chosen is the first child's output; all targets from children are
 * aggregated.
 */
public class ParallelPipeline extends CVPipeline<CVPipelineResult, ParallelPipelineSettings> {
    private final List<CVPipeline> children = new ArrayList<>();

    public ParallelPipeline() {
        super(FrameThresholdType.NONE);
        this.settings = new ParallelPipelineSettings();
    }

    public ParallelPipeline(ParallelPipelineSettings settings) {
        super(FrameThresholdType.NONE);
        this.settings = settings;
        instantiateChildren();
    }

    // Cache of last seen child settings to avoid recreating children each frame
    private java.util.List<CVPipelineSettings> lastSeenChildrenSettings = null;

    private void instantiateChildren() {
        children.clear();
        if (settings.children == null) return;
        for (var childSettings : settings.children) {
            if (childSettings == null) continue;
            switch (childSettings.pipelineType) {
                case Reflective ->
                        children.add(new ReflectivePipeline((ReflectivePipelineSettings) childSettings));
                case ColoredShape ->
                        children.add(new ColoredShapePipeline((ColoredShapePipelineSettings) childSettings));
                case AprilTag ->
                        children.add(new AprilTagPipeline((AprilTagPipelineSettings) childSettings));
                case Aruco -> children.add(new ArucoPipeline((ArucoPipelineSettings) childSettings));
                case ObjectDetection ->
                        children.add(
                                new ObjectDetectionPipeline((ObjectDetectionPipelineSettings) childSettings));
                case Sequential ->
                        children.add(new SequentialPipeline((SequentialPipelineSettings) childSettings));
                case Parallel ->
                        children.add(new ParallelPipeline((ParallelPipelineSettings) childSettings));
                default -> {
                    // Unsupported child type or built-in; skip
                }
            }
        }
    }

    private boolean childrenSettingsEqual(java.util.List<CVPipelineSettings> a, java.util.List<CVPipelineSettings> b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            CVPipelineSettings sa = a.get(i);
            CVPipelineSettings sb = b.get(i);
            if (sa == sb) continue;
            if (sa == null || sb == null) return false;
            if (sa.pipelineType != sb.pipelineType) return false;
            if (sa.pipelineNickname == null) {
                if (sb.pipelineNickname != null) return false;
            } else if (!sa.pipelineNickname.equals(sb.pipelineNickname)) return false;
        }
        return true;
    }

    @Override
    public CVPipeline getNodeAtPath(java.util.List<Integer> path) {
        if (path == null || path.isEmpty()) return this;
        int idx = path.get(0);
        if (idx < 0 || idx >= children.size()) return null;
        var child = children.get(idx);
        return child.getNodeAtPath(path.subList(1, path.size()));
    }

    @Override
    protected void setPipeParamsImpl() {
        if (!childrenSettingsEqual(lastSeenChildrenSettings, settings.children)) {
            instantiateChildren();
            lastSeenChildrenSettings = settings.children == null ? null : new java.util.ArrayList<>(settings.children);
        }
        for (int i = 0; i < children.size(); i++) {
            var childPipeline = children.get(i);
            var childSettings = settings.children.get(i);
            childPipeline.setPipeParams(frameStaticProperties, childSettings, cameraQuirks);
        }
    }

    @Override
    protected CVPipelineResult process(Frame frame, ParallelPipelineSettings settings) {
        double sumProcessingNanos = 0.0;
        double fps = 0.0;
        List<TrackedTarget> aggregatedTargets = new java.util.ArrayList<>();
        Optional<?> multiTagResult = Optional.empty();
        List<String> classNames = new ArrayList<>();

        CVPipelineResult firstResult = null;

        for (int i = 0; i < children.size(); i++) {
            var child = children.get(i);

            // Clone the frame for this child (shallow copy of mats)
            var clonedColor = new CVMat();
            var clonedProcessed = new CVMat();
            var clone =
                    new Frame(
                            frame.sequenceID,
                            clonedColor,
                            clonedProcessed,
                            frame.type,
                            frame.timestampNanos,
                            frame.frameStaticProperties);
            frame.copyTo(clone);

            CVPipelineResult childResult = child.run(clone, cameraQuirks);

            if (i == 0) {
                firstResult = childResult;
            } else {
                // release childResult frame ownership by releasing inputAndOutputFrame if it's not the
                // first
                if (childResult.inputAndOutputFrame != null) {
                    childResult.inputAndOutputFrame.release();
                }
            }

            if (childResult.targets != null) aggregatedTargets.addAll(childResult.targets);
            if (childResult.multiTagResult != null && childResult.multiTagResult.isPresent()) {
                if (multiTagResult.isEmpty()) multiTagResult = (Optional<?>) childResult.multiTagResult;
            }
            if (childResult.objectDetectionClassNames != null)
                classNames.addAll(childResult.objectDetectionClassNames);

            sumProcessingNanos += childResult.processingNanos;
            fps = childResult.fps;

            // release non-first results' other resources
            if (i != 0) {
                childResult.release();
            }
        }

        Frame finalFrame =
                (firstResult != null && firstResult.inputAndOutputFrame != null)
                        ? firstResult.inputAndOutputFrame
                        : frame;

        CVPipelineResult result =
                new CVPipelineResult(
                        frame.sequenceID,
                        sumProcessingNanos,
                        fps,
                        aggregatedTargets,
                        Optional.empty(),
                        finalFrame,
                        classNames);
        if (multiTagResult.isPresent()) {
            try {
                //noinspection unchecked
                result.multiTagResult = (Optional) multiTagResult;
            } catch (Exception ignored) {
            }
        }
        return result;
    }
}
