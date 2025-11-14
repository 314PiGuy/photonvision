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

package org.photonvision.jni;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.util.TestUtils;

/** Loader for the ONNX detector native libraries. */
public class OnnxDetectorJNI extends PhotonJNICommon {
    private static OnnxDetectorJNI instance;
    private boolean loaded;

    private OnnxDetectorJNI() {
        this.loaded = false;
    }

    public static synchronized OnnxDetectorJNI getInstance() {
        if (instance == null) {
            instance = new OnnxDetectorJNI();
        }
        return instance;
    }

    private static boolean resourceExists(String libraryName) {
        String mappedName = System.mapLibraryName(libraryName);
        String resourcePath =
                "/nativelibraries/" + Platform.getNativeLibraryFolderName() + "/" + mappedName;
        URL resource = OnnxDetectorJNI.class.getResource(resourcePath);
        return resource != null;
    }

    public static synchronized void forceLoad() throws IOException {
        TestUtils.loadLibraries();

        List<String> libraries = new ArrayList<>();
        if (resourceExists("onnxruntime_providers_shared")) {
            libraries.add("onnxruntime_providers_shared");
        }
        if (resourceExists("onnxruntime_providers_cpu")) {
            libraries.add("onnxruntime_providers_cpu");
        }
        libraries.add("onnxruntime");
        libraries.add("onnx_jni");

        forceLoad(getInstance(), OnnxDetectorJNI.class, libraries);
    }

    @Override
    public boolean isLoaded() {
        return loaded;
    }

    @Override
    public void setLoaded(boolean state) {
        loaded = state;
    }
}
