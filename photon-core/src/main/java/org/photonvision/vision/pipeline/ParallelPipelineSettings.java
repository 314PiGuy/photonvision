package org.photonvision.vision.pipeline;

import com.fasterxml.jackson.annotation.JsonTypeName;
import java.util.ArrayList;
import java.util.List;

@JsonTypeName("ParallelPipelineSettings")
public class ParallelPipelineSettings extends AdvancedPipelineSettings {
    public List<CVPipelineSettings> children = new ArrayList<>();

    public ParallelPipelineSettings() {
        super();
        pipelineType = PipelineType.Parallel;
    }
}
