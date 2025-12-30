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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.photonvision.vision.processes.VisionModuleChangeSubscriber.setProperty;

import java.util.ArrayList;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.photonvision.common.util.numbers.DoubleCouple;
import org.photonvision.common.util.numbers.IntegerCouple;

public class VisionModuleChangeSubscriberTest {
    enum TestEnum {
        VALUE1,
        VALUE2
    }

    static class TestClass {
        public TestEnum enumField;
        public DoubleCouple doubleCoupleField;
        public IntegerCouple integerCoupleField;
        public double doubleField;
        public int intField;
        public boolean booleanField;
        public String stringField;

        public TestClass() {
            enumField = TestEnum.VALUE1;
            doubleCoupleField = new DoubleCouple(0, 0);
            integerCoupleField = new IntegerCouple(0, 0);
            doubleField = 0;
            intField = 0;
            booleanField = false;
            stringField = "";
        }
    }

    @Test
    // Either set with the enum Variant or the ordinal value
    void testSetEnumField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(obj.enumField, TestEnum.VALUE1);

        setProperty(obj, "enumField", TestEnum.VALUE2.ordinal());
        assertEquals(TestEnum.VALUE2, obj.enumField);
    }

    @Test
    void testSetDoubleCoupleField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(new DoubleCouple(0, 0), obj.doubleCoupleField);

        ArrayList<Number> values = new ArrayList<>();
        values.add(1.1);
        values.add(2.2);

        setProperty(obj, "doubleCoupleField", values);

        assertEquals(1.1, obj.doubleCoupleField.getFirst());
        assertEquals(2.2, obj.doubleCoupleField.getSecond());
    }

    @Test
    void testSetIntegerCoupleField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(new IntegerCouple(0, 0), obj.integerCoupleField);

        ArrayList<Number> values = new ArrayList<>();
        values.add(1);
        values.add(2);

        setProperty(obj, "integerCoupleField", values);

        assertEquals(1, obj.integerCoupleField.getFirst());
        assertEquals(2, obj.integerCoupleField.getSecond());
    }

    @Test
    void testSetDoubleField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(0, obj.doubleField);

        setProperty(obj, "doubleField", 3.14);
        assertEquals(3.14, obj.doubleField);
    }

    @Test
    void testSetIntField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(0, obj.intField);

        setProperty(obj, "intField", 42);
        assertEquals(42, obj.intField);
    }

    @Test
    void testSetBooleanField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals(false, obj.booleanField);

        setProperty(obj, "booleanField", 1);
        assertTrue(obj.booleanField);

        setProperty(obj, "booleanField", 0);
        assertFalse(obj.booleanField);
    }

    @Test
    void testSetStringField() throws Exception {
        TestClass obj = new TestClass();
        assertEquals("", obj.stringField);

        setProperty(obj, "stringField", "test");
        assertEquals("test", obj.stringField);
    }

    @Test
    void testSetNonExistentField() {
        TestClass obj = new TestClass();
        Executable executable = () -> setProperty(obj, "nonExistentField", 1);
        assertThrows(NoSuchFieldException.class, executable);
    }

    @Test
    void testSetFieldWithIncompatibleType() {
        TestClass obj = new TestClass();
        Executable executable = () -> setProperty(obj, "doubleField", "string");
        assertThrows(Exception.class, executable);
    }

    @Test
    void testSetSourceCameraAndPipelineEnvelope() throws Exception {
        org.photonvision.vision.pipeline.SequentialPipelineSettings seq =
                new org.photonvision.vision.pipeline.SequentialPipelineSettings();

        java.util.Map<String, Object> payload = new java.util.HashMap<>();
        payload.put("children", new java.util.ArrayList<Object>());
        payload.put("type", "Parallel");

        // Set the sourceCamera via special-case
        setProperty(seq, "sourceCamera", "Cam1");
        assertEquals("Cam1", seq.sourceCamera);

        // Apply an envelope to set pipeline type and children
        setProperty(seq, "pipeline", payload);
        assertEquals(org.photonvision.vision.pipeline.PipelineType.Parallel, seq.pipelineType);
    }

    @Test
    void testResolveModelShorthand() throws Exception {
        // Build a payload similar to the UI-imported format
        var payload = new java.util.HashMap<String, Object>();
        var pipeline = new java.util.HashMap<String, Object>();
        var childPayload = new java.util.HashMap<String, Object>();
        var modelMap = new java.util.HashMap<String, Object>();
        modelMap.put("name", "yolo11n_640");
        modelMap.put("backend", "ONNX");
        childPayload.put("model", modelMap);
        var childWrapper = java.util.Arrays.asList("Object Detection", childPayload);
        pipeline.put("type", "Parallel");
        pipeline.put("children", java.util.Arrays.asList(childWrapper));
        payload.put("pipeline", pipeline);

        // Add a fake model property into the config so the resolver can find it
        var conf = new org.photonvision.common.configuration.PhotonConfiguration();
        var nnpm = new org.photonvision.common.configuration.NeuralNetworkPropertyManager();
        var labels = new java.util.LinkedList<String>();
        labels.add("person");
        var mp = new org.photonvision.common.configuration.NeuralNetworkPropertyManager.ModelProperties(
                java.nio.file.Path.of("/models/yolo11n_640.onnx"),
                "yolo11n_640",
                labels,
                640,
                640,
                org.photonvision.common.configuration.NeuralNetworkModelManager.Family.ONNX,
                org.photonvision.common.configuration.NeuralNetworkModelManager.Version.YOLOV11);
        nnpm.addModelProperties(mp);
        conf.setNeuralNetworkProperties(nnpm);

        // Inject the config into the SqlConfigProvider used by ConfigManager
        var cm = org.photonvision.common.configuration.ConfigManager.getInstance();
        // Use reflection to set the private provider.config to our test config
        java.lang.reflect.Field providerField = org.photonvision.common.configuration.ConfigManager.class.getDeclaredField("m_provider");
        providerField.setAccessible(true);
        var provider = providerField.get(cm);
        java.lang.reflect.Method setConfigMethod = provider.getClass().getMethod("setConfig", org.photonvision.common.configuration.PhotonConfiguration.class);
        setConfigMethod.invoke(provider, conf);

        // Run normalization
        org.photonvision.vision.processes.VisionModuleChangeSubscriber.normalizeModelShorthandInPipelinePayload(payload);

        // Verify that the nested model map now contains modelPath
        var children = ((java.util.Map) payload.get("pipeline")).get("children");
        var child = ((java.util.List) children).get(0);
        var inner = (java.util.List) child;
        var pl = (java.util.Map) inner.get(1);
        var model = (java.util.Map) pl.get("model");
        assertTrue(model.containsKey("modelPath"));
        assertEquals("/models/yolo11n_640.onnx", model.get("modelPath"));


    }
}
