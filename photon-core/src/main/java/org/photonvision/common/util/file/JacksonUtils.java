/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.common.util.file;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ext.NioPathDeserializer;
import com.fasterxml.jackson.databind.ext.NioPathSerializer;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.jsontype.BasicPolymorphicTypeValidator;
import com.fasterxml.jackson.databind.jsontype.PolymorphicTypeValidator;
import com.fasterxml.jackson.databind.module.SimpleModule;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import org.eclipse.jetty.io.EofException;

public class JacksonUtils {
    public static class UIMap extends HashMap<String, Object> {}

    // Custom Path key deserializer for Maps with Path keys
    public static class PathKeySerializer
            extends com.fasterxml.jackson.databind.JsonSerializer<Path> {
        @Override
        public void serialize(Path value, JsonGenerator gen, SerializerProvider serializers)
                throws IOException {
            if (value == null) {
                gen.writeNull();
            } else {
                gen.writeFieldName(value.toUri().toString());
            }
        }
    }

    // Custom Path key deserializer for Maps with Path keys
    public static class PathKeyDeserializer extends com.fasterxml.jackson.databind.KeyDeserializer {
        @Override
        public Object deserializeKey(String key, DeserializationContext ctxt) throws IOException {
            if (key == null || key.isEmpty()) {
                return null;
            }
            return Paths.get(URI.create(key));
        }
    }

    // Helper method to create ObjectMapper with Path serialization support
    private static ObjectMapper createObjectMapperWithPathSupport(Class<?> baseType) {
        PolymorphicTypeValidator ptv =
                BasicPolymorphicTypeValidator.builder().allowIfBaseType(baseType).build();

        SimpleModule pathModule = new SimpleModule();
        pathModule.addSerializer(Path.class, new NioPathSerializer());
        pathModule.addKeySerializer(Path.class, new PathKeySerializer());
        pathModule.addDeserializer(Path.class, new NioPathDeserializer());
        pathModule.addKeyDeserializer(Path.class, new PathKeyDeserializer());

        return JsonMapper.builder()
                .configure(JsonReadFeature.ALLOW_JAVA_COMMENTS, true)
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                .activateDefaultTyping(ptv, ObjectMapper.DefaultTyping.JAVA_LANG_OBJECT)
                .addModule(pathModule)
                .build();
    }

    // Helper method like createObjectMapperWithPathSupport but WITHOUT activating default typing.
    // This is useful for converting concrete types where Jackson's default typing can interfere
    // with direct conversions (e.g., classes that use WRAPPER_ARRAY polymorphic forms).
    private static ObjectMapper createPlainObjectMapperWithPathSupport() {
        SimpleModule pathModule = new SimpleModule();
        pathModule.addSerializer(Path.class, new NioPathSerializer());
        pathModule.addKeySerializer(Path.class, new PathKeySerializer());
        pathModule.addDeserializer(Path.class, new NioPathDeserializer());
        pathModule.addKeyDeserializer(Path.class, new PathKeyDeserializer());

        return JsonMapper.builder()
                .configure(JsonReadFeature.ALLOW_JAVA_COMMENTS, true)
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                .addModule(pathModule)
                .build();
    }

    public static <T> void serialize(Path path, T object) throws IOException {
        serialize(path, object, true);
    }

    public static <T> String serializeToString(T object) throws IOException {
        ObjectMapper objectMapper = createObjectMapperWithPathSupport(object.getClass());
        return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(object);
    }

    public static <T> void serialize(Path path, T object, boolean forceSync) throws IOException {
        ObjectMapper objectMapper = createObjectMapperWithPathSupport(object.getClass());
        String json = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(object);
        saveJsonString(json, path, forceSync);
    }

    public static <T> T deserialize(Map<?, ?> s, Class<T> ref) throws IOException {
        // Delegate to the generic object-based deserializer
        return deserialize((Object) s, ref);
    }

    public static <T> T deserialize(Object s, Class<T> ref) throws IOException {
        ObjectMapper objectMapper = createObjectMapperWithPathSupport(ref);

        if (s == null) {
            return null;
        }

        // Log the raw payload we're about to attempt to deserialize so we can capture
        // the exact inputs that cause downstream Jackson failures.
        try {
            String payloadLog = safeSerializeForLog(s);
            org.photonvision.common.logging.LoggingUtils.getLogger(JacksonUtils.class, "JacksonUtils")
                    .info("Deserializing to " + ref.getName() + " payload: " + payloadLog);
        } catch (Exception ignore) {
            try {
                org.photonvision.common.logging.LoggingUtils.getLogger(JacksonUtils.class, "JacksonUtils")
                        .info("Deserializing to " + ref.getName() + " payload: " + String.valueOf(s));
            } catch (Exception ignore2) {
                // best effort logging only
            }
        }

        // If it's already a JSON string, read directly
        if (s instanceof String) {
            String str = (String) s;
            if (str.length() == 0) {
                throw new EofException("Provided empty string for class " + ref.getName());
            }
            return objectMapper.readValue(str, ref);
        }

        // If it's a Map, attempt to determine type information first (e.g., the user
        // supplied an object form without the WRAPPER_ARRAY polymorphic type marker).
        if (s instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, ?> map = (Map<String, ?>) s;

            // Fast path: support a simple explicit envelope form from the UI:
            // { "type": "ObjectDetection", "payload": { ... } }
            if (map.containsKey("type") && map.containsKey("payload")) {
                Object typeObj = map.get("type");
                String typeStr = typeObj == null ? null : String.valueOf(typeObj);
                Object payload = map.get("payload");
                Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings> specific =
                        mapTypeNameToSettingsClass(typeStr);
                ObjectMapper plain = createPlainObjectMapperWithPathSupport();
                if (specific != null) {
                    try {
                        Object normalizedPayload = unwrapWrapperArrays(payload);
                        @SuppressWarnings("unchecked")
                        T converted = (T) plain.convertValue(normalizedPayload, specific);
                        return converted;
                    } catch (IllegalArgumentException e) {
                        try {
                            String json = plain.writeValueAsString(payload);
                            @SuppressWarnings("unchecked")
                            T converted = (T) plain.readValue(json, specific);
                            return converted;
                        } catch (Exception inner) {
                            // fall through to other strategies
                        }
                    }
                } else {
                    // If we don't recognize the type, attempt to deserialize payload
                    // into the requested ref type as a best-effort fallback.
                    try {
                        Object normalizedPayload = unwrapWrapperArrays(payload);
                        String json = objectMapper.writeValueAsString(normalizedPayload);
                        return objectMapper.readValue(json, ref);
                    } catch (Exception ignore) {
                        // fall through
                    }
                }
            }

            // Look for common keys that indicate the concrete pipeline type. If present,
            // map them to a concrete settings class and convert to that class to avoid
            // Jackson's WRAPPER_ARRAY expectations.
            Object typeVal = map.containsKey("type") ? map.get("type") : map.get("pipelineType");
            if (typeVal != null) {
                String typeStr = null;
                if (typeVal instanceof String) {
                    typeStr = (String) typeVal;
                } else if (typeVal instanceof Number) {
                    int n = ((Number) typeVal).intValue();
                    // Try to match by ordinal or by baseIndex used in PipelineType
                    for (org.photonvision.vision.pipeline.PipelineType pt :
                            org.photonvision.vision.pipeline.PipelineType.values()) {
                        if (pt.ordinal() == n || pt.baseIndex == n) {
                            typeStr = pt.name();
                            break;
                        }
                    }
                } else {
                    typeStr = String.valueOf(typeVal);
                }

                if (typeStr != null) {
                    Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings> specific =
                            mapTypeNameToSettingsClass(typeStr);
                    if (specific != null) {
                        // Use a plain mapper (no default typing) to avoid WRAPPER_ARRAY expectations
                        ObjectMapper plain = createPlainObjectMapperWithPathSupport();
                        try {
                            @SuppressWarnings("unchecked")
                            T converted = (T) plain.convertValue(map, specific);
                            return converted;
                        } catch (IllegalArgumentException e) {
                            try {
                                String json = plain.writeValueAsString(map);
                                @SuppressWarnings("unchecked")
                                T converted = (T) plain.readValue(json, specific);
                                return converted;
                            } catch (Exception inner) {
                                // fall through to try other strategies
                            }
                        }
                    }
                    // Log that we attempted object->specific mapping path
                    try {
                        final String typeStrFinal = typeStr;
                        org.photonvision.common.logging.LoggingUtils.getLogger(
                                        JacksonUtils.class, "JacksonUtils")
                                .trace(() -> "Attempted map-based type resolution for typeStr=" + typeStrFinal);
                    } catch (Exception ignore) {
                    }
                }
            }

            // First, try converting directly to the requested reference type. This
            // avoids eager attempts to materialize heavier pipeline settings classes
            // (which can have native dependencies) when the caller already knows the
            // desired target type.
            ObjectMapper plain = createPlainObjectMapperWithPathSupport();
            try {
                @SuppressWarnings("unchecked")
                T converted = (T) plain.convertValue(map, ref);
                return converted;
            } catch (IllegalArgumentException e) {
                try {
                    String json = plain.writeValueAsString(map);
                    @SuppressWarnings("unchecked")
                    T converted = (T) plain.readValue(json, ref);
                    return converted;
                } catch (Exception inner) {
                    // fall through to candidate trial conversion below
                }
            }

            // As a last ditch, attempt to convert the object into any known pipeline
            // settings class by trial. This helps when the incoming object is in
            // object-form but lacks explicit type fields.
            Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings>[] candidates =
                    new Class[] {
                        org.photonvision.vision.pipeline.ReflectivePipelineSettings.class,
                        org.photonvision.vision.pipeline.ColoredShapePipelineSettings.class,
                        org.photonvision.vision.pipeline.AprilTagPipelineSettings.class,
                        org.photonvision.vision.pipeline.ArucoPipelineSettings.class,
                        org.photonvision.vision.pipeline.ObjectDetectionPipelineSettings.class,
                        org.photonvision.vision.pipeline.SequentialPipelineSettings.class,
                        org.photonvision.vision.pipeline.ParallelPipelineSettings.class
                    };
            for (Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings> cand : candidates) {
                try {
                    @SuppressWarnings("unchecked")
                    T converted = (T) plain.convertValue(map, cand);
                    return converted;
                } catch (IllegalArgumentException e) {
                    try {
                        String json = plain.writeValueAsString(map);
                        @SuppressWarnings("unchecked")
                        T converted = (T) plain.readValue(json, cand);
                        return converted;
                    } catch (Exception inner) {
                        // Try next candidate
                    }
                }
            }
            try {
                org.photonvision.common.logging.LoggingUtils.getLogger(JacksonUtils.class, "JacksonUtils")
                        .trace(
                                () ->
                                        "Tried candidate conversions for map-based pipeline settings and none matched");
            } catch (Exception ignore) {
            }

            // As a final fallback, try wrapping the object in wrapper-array form with a
            // candidate type name and let the polymorphic deserializer pick the right one.
            String json = objectMapper.writeValueAsString(map);
            String[] candidateNames =
                    new String[] {
                        "Reflective",
                        "ColoredShape",
                        "AprilTag",
                        "Aruco",
                        "ObjectDetection",
                        "Sequential",
                        "Parallel"
                    };
            for (String cname : candidateNames) {
                String wrapped = "[" + objectMapper.writeValueAsString(cname) + "," + json + "]";
                try {
                    return objectMapper.readValue(wrapped, ref);
                } catch (Exception e) {
                    // Try next candidate
                }
            }

            // Fallback to the generic conversion if no type information worked.
            return objectMapper.convertValue(s, ref);
        }

        // If it's a collection/array (e.g. wrapper-array polymorphic form), handle wrapper arrays
        if (s instanceof java.util.Collection || s.getClass().isArray()) {
            // Normalize to a List for easy access
            java.util.List<?> list;
            if (s instanceof java.util.Collection) {
                list = new java.util.ArrayList<>((java.util.Collection<?>) s);
            } else {
                list = java.util.Arrays.asList((Object[]) s);
            }

            // Handle wrapper-array polymorphic form: ["TypeName", {...}]
            if (list.size() == 2 && list.get(0) instanceof String) {
                String typeName = (String) list.get(0);
                Object payload = list.get(1);

                // Attempt to map common short type names (e.g., "ObjectDetection", "AprilTag")
                // to their concrete settings classes. This makes the UI's shorthand names
                // compatible with Jackson's polymorphic typing.
                Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings> specific =
                        mapTypeNameToSettingsClass(typeName);
                if (specific != null) {
                    // Convert payload to the specific settings class first, which avoids
                    // Jackson trying to interpret the wrapper-array against the concrete
                    // settings class (which would fail since the input is still an array).
                    // Use the plain mapper (no default typing) so WRAPPER_ARRAY polymorphic
                    // expectations on the concrete class do not cause failures when the
                    // payload is provided as an object.
                    // If payload itself is a wrapper-array like ["java.util.LinkedHashMap", {...}],
                    // unwrap it to the inner object to allow normal object -> POJO conversion.
                    if (payload instanceof java.util.Collection
                            || payload != null && payload.getClass().isArray()) {
                        java.util.List<?> innerList;
                        if (payload instanceof java.util.Collection) {
                            innerList = new java.util.ArrayList<>((java.util.Collection<?>) payload);
                        } else {
                            innerList = java.util.Arrays.asList((Object[]) payload);
                        }
                        if (innerList.size() == 2 && innerList.get(1) instanceof java.util.Map) {
                            payload = innerList.get(1);
                            try {
                                org.photonvision.common.logging.LoggingUtils.getLogger(
                                                JacksonUtils.class, "JacksonUtils")
                                        .trace(() -> "Unwrapped nested wrapper-array payload for type " + typeName);
                            } catch (Exception ignore) {
                            }
                        }
                    }

                    ObjectMapper plain = createPlainObjectMapperWithPathSupport();
                    try {
                        @SuppressWarnings("unchecked")
                        T converted = (T) plain.convertValue(payload, specific);
                        // Log a debug message about the successful conversion when possible
                        try {
                            org.photonvision.common.logging.LoggingUtils.getLogger(
                                            JacksonUtils.class, "JacksonUtils")
                                    .trace(
                                            () ->
                                                    "Converted wrapper-array payload to specific class "
                                                            + specific.getSimpleName());
                        } catch (Exception ignore) {
                        }
                        return converted;
                    } catch (IllegalArgumentException e) {
                        // Try a safe serialize->read path which can handle some polymorphic edge cases
                        try {
                            String json = plain.writeValueAsString(payload);
                            @SuppressWarnings("unchecked")
                            T converted = (T) plain.readValue(json, specific);
                            return converted;
                        } catch (Exception inner) {
                            // fall through and allow outer logic to handle
                        }
                    }
                }
            }

            // Fallback: serialize entire structure and read as the requested type
            String json = objectMapper.writeValueAsString(s);
            return objectMapper.readValue(json, ref);
        }

        // If it's a Jackson JsonNode, handle array/object types
        if (s instanceof com.fasterxml.jackson.databind.JsonNode) {
            var node = (com.fasterxml.jackson.databind.JsonNode) s;
            if (node.isArray() || node.isObject()) {
                String json = objectMapper.writeValueAsString(node);
                return objectMapper.readValue(json, ref);
            }
        }

        // Fallback - attempt generic conversion
        return objectMapper.convertValue(s, ref);
    }

    // Map common human-friendly or shortened type names used in UI JSON to concrete settings classes
    public static Class<? extends org.photonvision.vision.pipeline.CVPipelineSettings>
            mapTypeNameToSettingsClass(String typeName) {
        String n = typeName == null ? "" : typeName.toLowerCase();
        if (n.contains("reflect"))
            return org.photonvision.vision.pipeline.ReflectivePipelineSettings.class;
        if (n.contains("colored") || n.contains("coloredshape"))
            return org.photonvision.vision.pipeline.ColoredShapePipelineSettings.class;
        if (n.contains("april")) return org.photonvision.vision.pipeline.AprilTagPipelineSettings.class;
        if (n.contains("aruco")) return org.photonvision.vision.pipeline.ArucoPipelineSettings.class;
        if (n.contains("object") || n.contains("detection"))
            return org.photonvision.vision.pipeline.ObjectDetectionPipelineSettings.class;
        if (n.contains("sequent"))
            return org.photonvision.vision.pipeline.SequentialPipelineSettings.class;
        if (n.contains("parallel"))
            return org.photonvision.vision.pipeline.ParallelPipelineSettings.class;
        return null;
    }

    // Recursively unwrap wrapper-array style objects produced by the UI/front-end.
    // For example: ["java.util.LinkedHashMap", { ... }] -> { ... }
    public static Object unwrapWrapperArrays(Object obj) {
        if (obj == null) return null;

        if (obj instanceof java.util.Collection || obj.getClass().isArray()) {
            java.util.List<?> list;
            if (obj instanceof java.util.Collection) {
                list = new java.util.ArrayList<>((java.util.Collection<?>) obj);
            } else {
                list = java.util.Arrays.asList((Object[]) obj);
            }
            if (list.size() == 2 && list.get(0) instanceof String) {
                // Unwrap and recurse into the payload element
                return unwrapWrapperArrays(list.get(1));
            }
            // Otherwise, recurse into each element
            java.util.List<Object> out = new java.util.ArrayList<>();
            for (Object o : list) out.add(unwrapWrapperArrays(o));
            return out;
        }

        if (obj instanceof java.util.Map) {
            java.util.Map<Object, Object> map = new java.util.LinkedHashMap<>();
            for (java.util.Map.Entry<?, ?> e : ((java.util.Map<?, ?>) obj).entrySet()) {
                map.put(e.getKey(), unwrapWrapperArrays(e.getValue()));
            }
            return map;
        }

        return obj;
    }

    // Safe helper for producing a string representation of a payload for logging.
    // Attempts JSON serialization first, then falls back to toString().
    private static String safeSerializeForLog(Object o) {
        if (o == null) return "null";
        try {
            return serializeToString(o);
        } catch (Exception e) {
            try {
                return String.valueOf(o);
            } catch (Exception inner) {
                return "<unserializable-payload>";
            }
        }
    }

    public static <T> T deserialize(String s, Class<T> ref) throws IOException {
        if (s.length() == 0) {
            throw new EofException("Provided empty string for class " + ref.getName());
        }

        ObjectMapper objectMapper = createObjectMapperWithPathSupport(ref);
        objectMapper.enable(DeserializationFeature.READ_UNKNOWN_ENUM_VALUES_AS_NULL);

        return objectMapper.readValue(s, ref);
    }

    public static <T> T deserialize(Path path, Class<T> ref) throws IOException {
        ObjectMapper objectMapper = createObjectMapperWithPathSupport(ref);
        File jsonFile = new File(path.toString());
        if (jsonFile.exists() && jsonFile.length() > 0) {
            // Read as an untyped object using the plain mapper (no default typing)
            // so that legacy files (object or array forms) are parsed without
            // forcing WRAPPER_ARRAY polymorphism.
            ObjectMapper plainReader = createPlainObjectMapperWithPathSupport();
            Object parsed = plainReader.readValue(jsonFile, Object.class);
            return deserialize(parsed, ref);
        }
        return null;
    }

    private static void saveJsonString(String json, Path path, boolean forceSync) throws IOException {
        var file = path.toFile();
        if (file.getParentFile() != null && !file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        if (!file.exists()) {
            if (!file.canWrite()) {
                file.setWritable(true);
            }
            file.createNewFile();
        }
        FileOutputStream fileOutputStream = new FileOutputStream(file);
        fileOutputStream.write(json.getBytes());
        fileOutputStream.flush();
        if (forceSync) {
            FileDescriptor fileDescriptor = fileOutputStream.getFD();
            fileDescriptor.sync();
        }
        fileOutputStream.close();
    }
}
