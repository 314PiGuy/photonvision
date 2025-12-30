package org.photonvision.common.util.file;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

public class JacksonUtilsTest {
    static class SimplePipeline extends org.photonvision.vision.pipeline.CVPipelineSettings {
        public double myValue = 0.0;
    }

    @Test
    void testDeserializeEnvelopeAsPayload() throws Exception {
        Map<String, Object> payload = new HashMap<>();
        payload.put("myValue", 7.5);

        // Pass the payload directly (no explicit 'type' wrapper)
        SimplePipeline settings = JacksonUtils.deserialize(payload, SimplePipeline.class);
        assertEquals(7.5, settings.myValue);
    }

    @Test
    void testUnwrapWrapperArrays() {
        Map<String, Object> payload = new HashMap<>();
        payload.put("myValue", 3.3);

        List<Object> wrapper = List.of("SimplePipeline", payload);
        Object unwrapped = JacksonUtils.unwrapWrapperArrays(wrapper);
        assertTrue(unwrapped instanceof java.util.Map);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> unwrappedMap = (java.util.Map<String, Object>) unwrapped;
        assertEquals(3.3, unwrappedMap.get("myValue"));

        List<Object> inner = List.of("java.util.LinkedHashMap", payload);
        List<Object> nested = List.of("SimplePipeline", inner);
        Object unwrappedNested = JacksonUtils.unwrapWrapperArrays(nested);
        assertTrue(unwrappedNested instanceof java.util.Map);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> nestedMap = (java.util.Map<String, Object>) unwrappedNested;
        assertEquals(3.3, nestedMap.get("myValue"));
    }

    @Test
    void testMapTypeNameToSettingsClass() {
        assertEquals(
                org.photonvision.vision.pipeline.ObjectDetectionPipelineSettings.class,
                JacksonUtils.mapTypeNameToSettingsClass("ObjectDetection"));
        assertEquals(
                org.photonvision.vision.pipeline.AprilTagPipelineSettings.class,
                JacksonUtils.mapTypeNameToSettingsClass("April Tag"));
        assertEquals(
                org.photonvision.vision.pipeline.SequentialPipelineSettings.class,
                JacksonUtils.mapTypeNameToSettingsClass("Sequential"));
    }
}
