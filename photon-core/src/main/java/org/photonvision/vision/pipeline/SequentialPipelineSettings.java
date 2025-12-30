package org.photonvision.vision.pipeline;

import com.fasterxml.jackson.annotation.JsonTypeName;
import java.util.ArrayList;
import java.util.List;

@JsonTypeName("SequentialPipelineSettings")
public class SequentialPipelineSettings extends AdvancedPipelineSettings {
    public List<CVPipelineSettings> children = new ArrayList<>();

    public SequentialPipelineSettings() {
        super();
        pipelineType = PipelineType.Sequential;
    }
}
