package org.photonvision.vision.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;

/**
 * Executes a list of child pipelines sequentially, passing the output frame of each to the next.
 */
public class SequentialPipeline extends CVPipeline<CVPipelineResult, SequentialPipelineSettings> {
    private final List<CVPipeline> children = new ArrayList<>();
    // Cache of last seen child settings to avoid recreating children each frame
    private java.util.List<CVPipelineSettings> lastSeenChildrenSettings = null;

    public SequentialPipeline() {
        super(FrameThresholdType.NONE);
        this.settings = new SequentialPipelineSettings();
    }

    public SequentialPipeline(SequentialPipelineSettings settings) {
        super(FrameThresholdType.NONE);
        this.settings = settings;
        instantiateChildrenIfNeeded();
    }

    private boolean childrenSettingsEqual(
            java.util.List<CVPipelineSettings> a, java.util.List<CVPipelineSettings> b) {
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
                    // Unsupported child type or built-in; log and skip
                }
            }
        }
    }

    private void instantiateChildrenIfNeeded() {
        if (!childrenSettingsEqual(lastSeenChildrenSettings, settings.children)) {
            instantiateChildren();
            lastSeenChildrenSettings =
                    settings.children == null ? null : new java.util.ArrayList<>(settings.children);
        }
    }

    @Override
    protected void setPipeParamsImpl() {
        // Ensure children are up-to-date only when settings change
        if (!childrenSettingsEqual(lastSeenChildrenSettings, settings.children)) {
            instantiateChildren();
            lastSeenChildrenSettings =
                    settings.children == null ? null : new java.util.ArrayList<>(settings.children);
        }

        for (int i = 0; i < children.size(); i++) {
            var childPipeline = children.get(i);
            var childSettings = settings.children.get(i);
            childPipeline.setPipeParams(frameStaticProperties, childSettings, cameraQuirks);
        }
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
    protected CVPipelineResult process(Frame frame, SequentialPipelineSettings settings) {
        Frame curFrame = frame;
        double sumProcessingNanos = 0.0;
        double fps = 0.0;
        List<TrackedTarget> aggregatedTargets = new java.util.ArrayList<>();
        Optional<?> multiTagResult = Optional.empty();
        List<String> classNames = new ArrayList<>();

        CVPipelineResult lastResult = null;

        for (int i = 0; i < children.size(); i++) {
            var child = children.get(i);
            CVPipelineResult childResult = child.run(curFrame, cameraQuirks);

            // Aggregate targets
            if (childResult.targets != null) aggregatedTargets.addAll(childResult.targets);
            if (childResult.multiTagResult != null && childResult.multiTagResult.isPresent()) {
                if (multiTagResult.isEmpty()) multiTagResult = (Optional<?>) childResult.multiTagResult;
            }
            if (childResult.objectDetectionClassNames != null)
                classNames.addAll(childResult.objectDetectionClassNames);

            sumProcessingNanos += childResult.processingNanos;
            fps = childResult.fps; // last child's fps

            // Prepare next frame
            if (lastResult != null) {
                // release the previous result (we no longer need it)
                lastResult.release();
            }
            lastResult = childResult;
            curFrame = childResult.inputAndOutputFrame;
        }

        // Final frame should be the frame from lastResult if any, otherwise original
        Frame finalFrame =
                (lastResult != null && lastResult.inputAndOutputFrame != null)
                        ? lastResult.inputAndOutputFrame
                        : frame;

        // Build final CVPipelineResult
        CVPipelineResult result =
                new CVPipelineResult(
                        frame.sequenceID,
                        sumProcessingNanos,
                        fps,
                        aggregatedTargets,
                        Optional.empty(),
                        finalFrame,
                        classNames);
        // propagate multiTag result if set
        if (multiTagResult.isPresent()) {
            try {
                //noinspection unchecked
                result.multiTagResult = (Optional) multiTagResult;
            } catch (Exception ignored) {
            }
        }

        // Don't release final child -- ownership transferred to result
        return result;
    }
}
