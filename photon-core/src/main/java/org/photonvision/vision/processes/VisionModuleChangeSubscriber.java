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

package org.photonvision.vision.processes;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.wpi.first.math.Pair;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.opencv.core.Point;
import org.photonvision.common.configuration.NeuralNetworkPropertyManager.ModelProperties;
import org.photonvision.common.dataflow.DataChangeService;
import org.photonvision.common.dataflow.DataChangeSubscriber;
import org.photonvision.common.dataflow.events.DataChangeEvent;
import org.photonvision.common.dataflow.events.IncomingWebSocketEvent;
import org.photonvision.common.dataflow.events.OutgoingUIEvent;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.file.JacksonUtils;
import org.photonvision.common.util.numbers.DoubleCouple;
import org.photonvision.common.util.numbers.IntegerCouple;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.pipeline.AdvancedPipelineSettings;
import org.photonvision.vision.pipeline.CVPipelineSettings;
import org.photonvision.vision.pipeline.PipelineType;
import org.photonvision.vision.pipeline.UICalibrationData;
import org.photonvision.vision.target.RobotOffsetPointOperation;

@SuppressWarnings("unchecked")
public class VisionModuleChangeSubscriber extends DataChangeSubscriber {
    private final VisionModule parentModule;
    private final Logger logger;
    private List<VisionModuleChange<?>> settingChanges = new ArrayList<>();
    private final ReentrantLock changeListLock = new ReentrantLock();

    public VisionModuleChangeSubscriber(VisionModule parentModule) {
        this.parentModule = parentModule;
        logger =
                new Logger(
                        VisionModuleChangeSubscriber.class,
                        parentModule.visionSource.getSettables().getConfiguration().nickname,
                        LogGroup.VisionModule);
    }

    @Override
    public void onDataChangeEvent(DataChangeEvent<?> event) {
        // Camera index -1 means a "multicast event" (i.e. the event is received by all
        // cameras)
        if (event instanceof IncomingWebSocketEvent wsEvent
                && wsEvent.cameraUniqueName != null
                && wsEvent.cameraUniqueName.equals(parentModule.uniqueName())) {
            logger.trace("Got PSC event - propName: " + wsEvent.propertyName);
            changeListLock.lock();
            try {
                getSettingChanges()
                        .add(
                                new VisionModuleChange(
                                        wsEvent.propertyName,
                                        wsEvent.data,
                                        parentModule.pipelineManager.getCurrentPipeline().getSettings(),
                                        wsEvent.originContext));
            } finally {
                changeListLock.unlock();
            }
        }
    }

    public List<VisionModuleChange<?>> getSettingChanges() {
        return settingChanges;
    }

    public void processSettingChanges() {
        // special case for non-PipelineSetting changes
        changeListLock.lock();
        try {
            for (var change : settingChanges) {
                var propName = change.getPropName();
                var newPropValue = change.getNewPropValue();
                var currentSettings = change.getCurrentSettings();
                var originContext = change.getOriginContext();
                switch (propName) {
                    case "pipelineName" -> newPipelineNickname((String) newPropValue);
                    case "newPipelineInfo" -> newPipelineInfo((Pair<String, PipelineType>) newPropValue);
                    case "deleteCurrPipeline" -> deleteCurrPipeline();
                    case "changePipeline" -> changePipeline((Integer) newPropValue);
                    case "startCalibration" -> startCalibration((Map<String, Object>) newPropValue);
                    case "requestNodeFrame" -> {
                        requestNodeFrame((Map<String, Object>) newPropValue);
                    }
                    case "saveInputSnapshot" -> parentModule.saveInputSnapshot();
                    case "saveOutputSnapshot" -> parentModule.saveOutputSnapshot();
                    case "takeCalSnapshot" -> parentModule.takeCalibrationSnapshot();
                    case "duplicatePipeline" -> duplicatePipeline((Integer) newPropValue);
                    case "calibrationUploaded" -> {
                        if (newPropValue instanceof CameraCalibrationCoefficients newCal) {
                            parentModule.addCalibrationToConfig(newCal);
                        } else {
                            logger.warn("Received invalid calibration data");
                        }
                    }
                    case "robotOffsetPoint" -> {
                        if (currentSettings instanceof AdvancedPipelineSettings curAdvSettings) {
                            robotOffsetPoint(curAdvSettings, (Integer) newPropValue);
                        }
                    }
                    case "changePipelineType" -> {
                        logger.info(
                                "Processing changePipelineType request: "
                                        + newPropValue
                                        + " for "
                                        + parentModule.uniqueName());
                        logger.info("Current settings pipelineType=" + currentSettings.pipelineType);
                        parentModule.changePipelineType((Integer) newPropValue);
                        logger.info(
                                "After change, pipelineType="
                                        + parentModule.pipelineManager.getCurrentPipelineSettings().pipelineType);
                        parentModule.saveAndBroadcastAll();
                    }
                    case "importPipeline" -> {
                        importPipeline(newPropValue);
                    }
                    case "isDriverMode" -> parentModule.setDriverMode((Boolean) newPropValue);
                    default -> {
                        // special case for camera settables
                        if (propName.startsWith("camera")) {
                            var propMethodName = "set" + propName.replace("camera", "");
                            var methods = parentModule.visionSource.getSettables().getClass().getMethods();
                            for (var method : methods) {
                                if (method.getName().equalsIgnoreCase(propMethodName)) {
                                    try {
                                        method.invoke(parentModule.visionSource.getSettables(), newPropValue);
                                    } catch (Exception e) {
                                        logger.error("Failed to invoke camera settable method: " + method.getName(), e);
                                    }
                                }
                            }
                        }

                        try {
                            setProperty(currentSettings, propName, newPropValue);
                            logger.trace("Set prop " + propName + " to value " + newPropValue);
                        } catch (NoSuchFieldException | IllegalAccessException e) {
                            logger.error(
                                    "Could not set prop "
                                            + propName
                                            + " with value "
                                            + newPropValue
                                            + " on "
                                            + currentSettings
                                            + " | "
                                            + e.getClass().getSimpleName(),
                                    e);
                        } catch (Exception e) {
                            logger.error("Unknown exception when setting PSC prop!", e);
                        }

                        parentModule.saveAndBroadcastSelective(originContext, propName, newPropValue);
                    }
                }
            }
            getSettingChanges().clear();
        } finally {
            changeListLock.unlock();
        }
    }

    public void newPipelineNickname(String newNickname) {
        logger.info("Changing pipeline nickname to " + newNickname);
        parentModule.pipelineManager.getCurrentPipelineSettings().pipelineNickname = newNickname;
        parentModule.saveAndBroadcastAll();
    }

    public void newPipelineInfo(Pair<String, PipelineType> typeName) {
        var type = typeName.getSecond();
        var name = typeName.getFirst();

        logger.info("Adding a " + type + " pipeline with name " + name);

        var addedSettings = parentModule.pipelineManager.addPipeline(type);
        addedSettings.pipelineNickname = name;
        parentModule.saveAndBroadcastAll();
    }

    public void deleteCurrPipeline() {
        var indexToDelete = parentModule.pipelineManager.getRequestedIndex();
        logger.info("Deleting current pipe at index " + indexToDelete);
        int newIndex = parentModule.pipelineManager.removePipeline(indexToDelete);
        parentModule.setPipeline(newIndex);
        parentModule.saveAndBroadcastAll();
    }

    public void changePipeline(int index) {
        if (index == parentModule.pipelineManager.getRequestedIndex()) {
            logger.debug("Skipping pipeline change, index " + index + " already active");
            return;
        }
        parentModule.setPipeline(index);
        parentModule.saveAndBroadcastAll();
    }

    public void startCalibration(Map<String, Object> data) {
        try {
            var deserialized = JacksonUtils.deserialize(data, UICalibrationData.class);
            parentModule.startCalibration(deserialized);
            parentModule.saveAndBroadcastAll();
        } catch (Exception e) {
            logger.error("Error deserializing start-calibration request", e);
        }
    }

    public void duplicatePipeline(int index) {
        var newIndex = parentModule.pipelineManager.duplicatePipeline(index);
        parentModule.setPipeline(newIndex);
        parentModule.saveAndBroadcastAll();
    }

    public void requestNodeFrame(Map<String, Object> data) {
        try {
            logger.info("Node frame request: " + data);
            // Expected payload: { "path": [0, 1, ...] }
            Object pathObj = data.get("path");

            var resp = new HashMap<String, Object>();
            resp.put("path", pathObj);

            if (!(pathObj instanceof java.util.List)) {
                resp.put("status", "invalid_path");
                DataChangeService.getInstance().publishEvent(OutgoingUIEvent.wrappedOf("nodeFrame", resp));
                return;
            }

            @SuppressWarnings("unchecked")
            java.util.List<Integer> path = new java.util.ArrayList<>();
            for (Object o : (java.util.List<Object>) pathObj) {
                if (o instanceof Number) path.add(((Number) o).intValue());
            }

            var current = parentModule.pipelineManager.getCurrentPipeline();
            var node = current.getNodeAtPath(path);
            if (node == null) {
                resp.put("status", "not_found");
                DataChangeService.getInstance().publishEvent(OutgoingUIEvent.wrappedOf("nodeFrame", resp));
                return;
            }

            var maybeB64 = node.getDebugImageBase64();
            if (maybeB64.isPresent()) {
                resp.put("status", "ok");
                resp.put("image_base64", maybeB64.get());
            } else {
                resp.put("status", "no_frame");
            }

            DataChangeService.getInstance().publishEvent(OutgoingUIEvent.wrappedOf("nodeFrame", resp));
        } catch (Exception e) {
            logger.error("Failed to handle node frame request", e);
        }
    }

    /**
     * Normalize pipeline import payloads from the UI by resolving shorthand model references
     * (e.g. { model: { name: "yolo11n_640", backend: "ONNX" } }) to a full
     * ModelProperties-like map that contains modelPath so server can resolve it correctly.
     */
    public static void normalizeModelShorthandInPipelinePayload(Object data) {
        try {
            if (data instanceof java.util.Map m) {
                Object pipelineObj = m.get("pipeline");
                if (pipelineObj instanceof java.util.Map pm) {
                    Object children = pm.get("children");
                    if (children instanceof java.util.List) {
                        for (int i = 0; i < ((java.util.List<?>) children).size(); i++) {
                            Object child = ((java.util.List<?>) children).get(i);
                            java.util.Map<?, ?> payload = null;

                            if (child instanceof java.util.Collection || child != null && child.getClass().isArray()) {
                                java.util.List<?> rawList;
                                if (child instanceof java.util.Collection) rawList = new java.util.ArrayList<>((java.util.Collection<?>) child);
                                else rawList = java.util.Arrays.asList((Object[]) child);
                                if (rawList.size() == 2 && rawList.get(1) instanceof java.util.Map) {
                                    payload = (java.util.Map<?, ?>) rawList.get(1);
                                }
                            } else if (child instanceof java.util.Map) {
                                payload = (java.util.Map<?, ?>) child;
                            }

                            if (payload != null) {
                                Object modelObj = payload.get("model");
                                if (modelObj instanceof java.util.Map) {
                                    java.util.Map<String, Object> modelMap = new java.util.HashMap<>((java.util.Map<String, Object>) modelObj);
                                    if (!modelMap.containsKey("modelPath") && modelMap.containsKey("name")) {
                                        String name = String.valueOf(modelMap.get("name"));
                                        String backend = modelMap.containsKey("backend") ? String.valueOf(modelMap.get("backend")) : null;
                                        var nnProps = org.photonvision.common.configuration.ConfigManager.getInstance().getConfig() != null ? org.photonvision.common.configuration.ConfigManager.getInstance().getConfig().neuralNetworkPropertyManager() : null;
                                        if (nnProps != null) {
                                            for (var candidate : nnProps.getModels()) {
                                                boolean backendMatches = true;
                                                if (backend != null) {
                                                    try {
                                                        backendMatches = candidate.family().name().equalsIgnoreCase(backend);
                                                    } catch (Exception ignored) {
                                                        backendMatches = false;
                                                    }
                                                }
                                                if (!backendMatches) continue;
                                                if (candidate.nickname() != null && candidate.nickname().equalsIgnoreCase(name)
                                                        || (candidate.modelPath() != null && candidate.modelPath().getFileName().toString().contains(name))) {
                                                    modelMap.put("modelPath", candidate.modelPath().toString());
                                                    // Update the payload map in-place
                                                    ((java.util.Map) payload).put("model", modelMap);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception ignore) {
        }
    }

    public void importPipeline(Object data) {
        try {
            // Pre-normalize shorthand model references in the incoming payload
            normalizeModelShorthandInPipelinePayload(data);

            // Convert incoming arbitrary JSON to a CVPipelineSettings instance
            var imported =
                    org.photonvision.common.util.file.JacksonUtils.deserialize(
                            data, CVPipelineSettings.class);

            // If there is a current user pipeline selected, replace its settings. Otherwise add as new
            // pipeline.
            int idx = parentModule.pipelineManager.getRequestedIndex();
            if (idx >= 0) {
                logger.info(
                        "Replacing pipeline at index "
                                + idx
                                + " with imported pipeline "
                                + imported.pipelineNickname);
                imported.pipelineIndex = idx;
                parentModule.pipelineManager.userPipelineSettings.set(idx, imported);
                // Ensure the vision module applies the new pipeline immediately
                parentModule.setPipeline(idx);
            } else {
                logger.info("Adding imported pipeline " + imported.pipelineNickname + " as new pipeline");
                parentModule.pipelineManager.addPipeline(imported);
                // Set the newly added pipeline as the active one
                int newIdx = parentModule.pipelineManager.userPipelineSettings.size() - 1;
                parentModule.setPipeline(newIdx);
            }

            parentModule.saveAndBroadcastAll();
            logger.info("Imported pipeline " + imported.pipelineNickname + " successfully");
        } catch (Exception e) {
            logger.error("Failed to import pipeline from JSON", e);
        }
    }

    public void robotOffsetPoint(AdvancedPipelineSettings curAdvSettings, int offsetIndex) {
        RobotOffsetPointOperation offsetOperation = RobotOffsetPointOperation.fromIndex(offsetIndex);

        var latestTarget = parentModule.lastPipelineResultBestTarget;
        if (latestTarget == null) {
            return;
        }

        var newPoint = latestTarget.getTargetOffsetPoint();
        switch (curAdvSettings.offsetRobotOffsetMode) {
            case Single -> {
                switch (offsetOperation) {
                    case CLEAR -> curAdvSettings.offsetSinglePoint = new Point();
                    case TAKE_SINGLE -> curAdvSettings.offsetSinglePoint = newPoint;
                    case TAKE_FIRST_DUAL, TAKE_SECOND_DUAL ->
                            logger.warn("Dual point operation in single point mode");
                }
            }
            case Dual -> {
                switch (offsetOperation) {
                    case CLEAR -> {
                        curAdvSettings.offsetDualPointA = new Point();
                        curAdvSettings.offsetDualPointAArea = 0;
                        curAdvSettings.offsetDualPointB = new Point();
                        curAdvSettings.offsetDualPointBArea = 0;
                    }
                    case TAKE_FIRST_DUAL -> {
                        // update point and area
                        curAdvSettings.offsetDualPointA = newPoint;
                        curAdvSettings.offsetDualPointAArea = latestTarget.getArea();
                    }
                    case TAKE_SECOND_DUAL -> {
                        // update point and area
                        curAdvSettings.offsetDualPointB = newPoint;
                        curAdvSettings.offsetDualPointBArea = latestTarget.getArea();
                    }
                    case TAKE_SINGLE -> logger.warn("Single point operation in dual point mode");
                }
            }
            case None -> logger.warn("Robot offset point operation requested, but no offset mode set");
        }
    }

    /**
     * Sets the value of a property in the given object using reflection. This method should not be
     * used generally and is only known to be correct in the context of `onDataChangeEvent`.
     *
     * @param currentSettings The object whose property needs to be set.
     * @param propName The name of the property to be set.
     * @param newPropValue The new value to be assigned to the property.
     * @throws IllegalAccessException If the field cannot be accessed.
     * @throws NoSuchFieldException If the field does not exist.
     * @throws Exception If an some other unknown exception occurs while setting the property.
     */
    protected static void setProperty(Object currentSettings, String propName, Object newPropValue)
            throws IllegalAccessException, NoSuchFieldException, Exception {
        java.lang.reflect.Field propField = null;
        try {
            propField = currentSettings.getClass().getField(propName);
        } catch (NoSuchFieldException nsfe) {
            // Special-case property names that aren't direct fields on CVPipelineSettings
            if ("pipeline".equals(propName) && newPropValue instanceof java.util.Map) {
                // Reuse the pipeline-envelope handling: set pipeline type and/or children
                var map = (java.util.Map<?, ?>) newPropValue;
                // Map type to pipelineType field if present
                Object typeVal = map.get("type");
                if (typeVal != null) {
                    try {
                        if (typeVal instanceof String s) {
                            for (org.photonvision.vision.pipeline.PipelineType pt :
                                    org.photonvision.vision.pipeline.PipelineType.values()) {
                                if (pt.name().equalsIgnoreCase(s)
                                        || pt.name().toLowerCase().contains(s.toLowerCase())) {
                                    try {
                                        var field = currentSettings.getClass().getField("pipelineType");
                                        field.set(currentSettings, pt);
                                    } catch (Exception ignore) {
                                    }
                                    break;
                                }
                            }
                        } else if (typeVal instanceof Number n) {
                            int ni = ((Number) n).intValue();
                            for (org.photonvision.vision.pipeline.PipelineType pt :
                                    org.photonvision.vision.pipeline.PipelineType.values()) {
                                if (pt.ordinal() == ni || pt.baseIndex == ni) {
                                    try {
                                        var field = currentSettings.getClass().getField("pipelineType");
                                        field.set(currentSettings, pt);
                                    } catch (Exception ignore) {
                                    }
                                    break;
                                }
                            }
                        }
                    } catch (Exception ignore) {
                    }
                }
                Object childrenObj = map.get("children");
                if (childrenObj == null && map.get("payload") instanceof java.util.Map) {
                    childrenObj = ((java.util.Map<?, ?>) map.get("payload")).get("children");
                }
                if (childrenObj instanceof java.util.List) {
                    setProperty(currentSettings, "children", childrenObj);
                }
                return;
            }

            if ("sourceCamera".equals(propName) && newPropValue instanceof String) {
                if (currentSettings instanceof org.photonvision.vision.pipeline.CVPipelineSettings cs) {
                    cs.sourceCamera = (String) newPropValue;
                }
                return;
            }

            // If we don't handle this special prop, rethrow so callers see the original error
            throw nsfe;
        }
        var propType = propField.getType();

        if (propType.isEnum()) {
            var actual = propType.getEnumConstants()[(int) newPropValue];
            propField.set(currentSettings, actual);
        } else if (propType.isAssignableFrom(DoubleCouple.class)) {
            var orig = (ArrayList<Number>) newPropValue;
            var actual = new DoubleCouple(orig.get(0), orig.get(1));
            propField.set(currentSettings, actual);
        } else if (propType.isAssignableFrom(IntegerCouple.class)) {
            var orig = (ArrayList<Number>) newPropValue;
            var actual = new IntegerCouple(orig.get(0).intValue(), orig.get(1).intValue());
            propField.set(currentSettings, actual);
        } else if (propType.equals(Double.TYPE)) {
            propField.setDouble(currentSettings, ((Number) newPropValue).doubleValue());
        } else if (propType.equals(Integer.TYPE)) {
            propField.setInt(currentSettings, (Integer) newPropValue);
        } else if (propType.equals(Boolean.TYPE)) {
            if (newPropValue instanceof Integer intValue) {
                propField.setBoolean(currentSettings, intValue != 0);
            } else {
                propField.setBoolean(currentSettings, (Boolean) newPropValue);
            }
        } else if (propField.getType() == ModelProperties.class
                && newPropValue instanceof LinkedHashMap) {
            ObjectMapper mapper = new ObjectMapper();
            ModelProperties modelProps = mapper.convertValue(newPropValue, ModelProperties.class);
            // If the incoming payload didn't contain a modelPath (e.g., short form { name: "yolo11n_640", backend: "ONNX" }),
            // attempt to resolve it from the configured models list by matching nickname or filename and backend.
            if (modelProps.modelPath() == null) {
                try {
                    Object nameObj = ((LinkedHashMap<?, ?>) newPropValue).get("name");
                    Object backendObj = ((LinkedHashMap<?, ?>) newPropValue).get("backend");
                    String name = nameObj instanceof String ? (String) nameObj : null;
                    String backend = backendObj instanceof String ? (String) backendObj : null;
                    if (name != null) {
                        var nnProps = org.photonvision.common.configuration.ConfigManager.getInstance()
                                .getConfig()
                                .neuralNetworkPropertyManager();
                        for (var candidate : nnProps.getModels()) {
                            boolean backendMatches = true;
                            if (backend != null) {
                                try {
                                    backendMatches = candidate.family().name().equalsIgnoreCase(backend);
                                } catch (Exception ignored) {
                                    backendMatches = false;
                                }
                            }
                            if (!backendMatches) continue;
                            if (candidate.nickname() != null && candidate.nickname().equalsIgnoreCase(name)) {
                                propField.set(currentSettings, candidate);
                                org.photonvision.common.logging.LoggingUtils.getLogger(VisionModuleChangeSubscriber.class, "").info("Resolved model shorthand '" + name + "' to path " + candidate.modelPath());
                                return;
                            }
                            if (candidate.modelPath() != null
                                    && (candidate.modelPath().getFileName().toString().contains(name)
                                            || candidate.modelPath().toString().contains(name))) {
                                propField.set(currentSettings, candidate);
                                org.photonvision.common.logging.LoggingUtils.getLogger(VisionModuleChangeSubscriber.class, "").info("Resolved model shorthand '" + name + "' to path " + candidate.modelPath());
                                return;
                            }
                        }
                    }
                } catch (Exception e) {
                    org.photonvision.common.logging.LoggingUtils.getLogger(VisionModuleChangeSubscriber.class, "").warn("Failed to resolve model shorthand to full ModelProperties: " + e.getMessage());
                }
            }
            propField.set(currentSettings, modelProps);
        } else if ("pipeline".equals(propName) && newPropValue instanceof java.util.Map) {
            // Support envelope form: { type: 'parallel', children: [...] }
            var map = (java.util.Map<?, ?>) newPropValue;
            // Map type string/number to PipelineType if provided
            Object typeVal = map.get("type");
            if (typeVal != null) {
                try {
                    if (typeVal instanceof String s) {
                        for (org.photonvision.vision.pipeline.PipelineType pt :
                                org.photonvision.vision.pipeline.PipelineType.values()) {
                            if (pt.name().equalsIgnoreCase(s)
                                    || pt.name().toLowerCase().contains(s.toLowerCase())) {
                                var field = currentSettings.getClass().getField("pipelineType");
                                field.set(currentSettings, pt);
                                break;
                            }
                        }
                    } else if (typeVal instanceof Number n) {
                        int ni = ((Number) n).intValue();
                        for (org.photonvision.vision.pipeline.PipelineType pt :
                                org.photonvision.vision.pipeline.PipelineType.values()) {
                            if (pt.ordinal() == ni || pt.baseIndex == ni) {
                                var field = currentSettings.getClass().getField("pipelineType");
                                field.set(currentSettings, pt);
                                break;
                            }
                        }
                    }
                } catch (Exception ignore) {
                }
            }
            Object childrenObj = map.get("children");
            if (childrenObj == null && map.get("payload") instanceof java.util.Map) {
                childrenObj = ((java.util.Map<?, ?>) map.get("payload")).get("children");
            }
            if (childrenObj instanceof java.util.List) {
                setProperty(currentSettings, "children", childrenObj);
            } else {
                // Fallback: try to deserialize map into CVPipelineSettings and copy over nickname
                try {
                    var converted =
                            org.photonvision.common.util.file.JacksonUtils.deserialize(
                                    map, org.photonvision.vision.pipeline.CVPipelineSettings.class);
                    if (converted != null
                            && converted.pipelineNickname != null
                            && currentSettings
                                    instanceof org.photonvision.vision.pipeline.CVPipelineSettings cs) {
                        cs.pipelineNickname = converted.pipelineNickname;
                    }
                } catch (Exception ignore) {
                }
            }
        } else if ("children".equals(propName) && newPropValue instanceof java.util.List) {
            // Convert arbitrary list elements (possibly Maps or wrapper arrays) to CVPipelineSettings
            // instances
            var list = (java.util.List<?>) newPropValue;
            var converted =
                    new java.util.ArrayList<org.photonvision.vision.pipeline.CVPipelineSettings>();
            try {
                for (var elem : list) {
                    try {
                        // Log the raw child payload to help capture the exact shape
                        String serializedChild;
                        try {
                            serializedChild =
                                    org.photonvision.common.util.file.JacksonUtils.serializeToString(elem);
                        } catch (Exception serEx) {
                            serializedChild = String.valueOf(elem);
                        }
                        org.photonvision.common.logging.LoggingUtils.getLogger(
                                        VisionModuleChangeSubscriber.class, "")
                                .info("Child element payload: " + serializedChild);

                        // If the element is a wrapper-array like ["TypeName", payload], prefer to
                        // map the type name to a concrete settings class and deserialize directly
                        // to that class to avoid polymorphic WRAPPER_ARRAY mismatches.
                        Object candidate = elem;
                        if (elem instanceof java.util.Collection || elem != null && elem.getClass().isArray()) {
                            java.util.List<?> rawList;
                            if (elem instanceof java.util.Collection)
                                rawList = new java.util.ArrayList<>((java.util.Collection<?>) elem);
                            else rawList = java.util.Arrays.asList((Object[]) elem);
                            if (rawList.size() == 2 && rawList.get(0) instanceof String) {
                                String typeName = (String) rawList.get(0);
                                var specific =
                                        org.photonvision.common.util.file.JacksonUtils.mapTypeNameToSettingsClass(
                                                typeName);
                                if (specific != null) {
                                    Object payload = rawList.get(1);
                                    var normalizedPayload =
                                            org.photonvision.common.util.file.JacksonUtils.unwrapWrapperArrays(payload);
                                    var convertedElem =
                                            org.photonvision.common.util.file.JacksonUtils.deserialize(
                                                    normalizedPayload, specific);
                                    converted.add(convertedElem);
                                    continue;
                                }
                            }
                        }

                        var normalized =
                                org.photonvision.common.util.file.JacksonUtils.unwrapWrapperArrays(elem);
                        var convertedElem =
                                org.photonvision.common.util.file.JacksonUtils.deserialize(
                                        normalized, org.photonvision.vision.pipeline.CVPipelineSettings.class);
                        converted.add(convertedElem);
                    } catch (Exception elemEx) {
                        try {
                            String serialized =
                                    org.photonvision.common.util.file.JacksonUtils.serializeToString(elem);
                            org.photonvision.common.logging.LoggingUtils.getLogger(
                                            VisionModuleChangeSubscriber.class, "")
                                    .error("Failed to convert children element: " + serialized, elemEx);
                        } catch (Exception serEx) {
                            org.photonvision.common.logging.LoggingUtils.getLogger(
                                            VisionModuleChangeSubscriber.class, "")
                                    .error("Failed to convert children element and failed to serialize it", elemEx);
                        }
                        throw elemEx;
                    }
                }
                propField.set(currentSettings, converted);
            } catch (Exception e) {
                throw new RuntimeException("Failed to convert children list to pipeline settings", e);
            }
        } else {
            propField.set(currentSettings, newPropValue);
        }
    }
}
