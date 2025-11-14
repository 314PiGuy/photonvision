# PhotonVision ONNX JNI

This module contains the cross-platform JNI bindings for running PhotonVision object detection models with the ONNX Runtime. The Java sources are packaged with the native binaries that are produced by the accompanying CMake project.

## Building the native library

```bash
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake_build
cmake --build cmake_build --target install -- -j 4
```

The install step copies the generated shared libraries into `cmake_build/lib`. Running `./gradlew :onnx_jni:build` afterwards will bundle the native libraries into the module jar under `nativelibraries/<platform>` so they can be loaded dynamically at runtime.
