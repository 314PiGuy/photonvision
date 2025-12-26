cmake_minimum_required(VERSION 3.16)

project(onnx_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DIRECTML "Use DirectML-enabled ONNX Runtime on Windows (nuget package)" ON)
set(ONNXRUNTIME_URL_OVERRIDE "" CACHE STRING "Override ONNX Runtime download URL")

option(ENABLE_ASAN "Enable address sanitizer" OFF)
if(ENABLE_ASAN)
    message(STATUS "Building with AddressSanitizer enabled")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined")
endif()

set(BUILD_SHARED_LIBS ON)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

set(ONNXRUNTIME_VERSION "1.19.2")
set(ONNXRUNTIME_URL "")
set(ONNXRUNTIME_ARCHIVE "")
set(ONNXRUNTIME_IS_NUGET OFF)

if(NOT ONNXRUNTIME_URL_OVERRIDE STREQUAL "")
    set(ONNXRUNTIME_URL "${ONNXRUNTIME_URL_OVERRIDE}")
    set(ONNXRUNTIME_FLAVOR "CustomOverride")
elseif(WIN32)
    if(USE_DIRECTML)
        # nuget for winml because the bums didnt put it on github releases
        set(ONNXRUNTIME_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ONNXRUNTIME_VERSION}")
        set(ONNXRUNTIME_FLAVOR "DirectML")
        set(ONNXRUNTIME_IS_NUGET ON)
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        set(ONNXRUNTIME_FLAVOR "CPU")
    endif()
elseif(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime")
endif()

if(ONNXRUNTIME_URL STREQUAL "")
    message(FATAL_ERROR "Could not determine ONNX Runtime download URL")
endif()
if(WIN32)
    message(STATUS "Using ONNX Runtime ${ONNXRUNTIME_VERSION} (${ONNXRUNTIME_FLAVOR}) from ${ONNXRUNTIME_URL}")
endif()

include(FetchContent)

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

if(ONNXRUNTIME_IS_NUGET)
    # NuGet layout for Microsoft.ML.OnnxRuntime.DirectML
    set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/build/native/include")
    set(ONNXRUNTIME_LIB_DIR "${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native")
else()
    set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
    set(ONNXRUNTIME_LIB_DIR "${onnxruntime_SOURCE_DIR}/lib")
endif()

file(GLOB ONNXRUNTIME_SHARED_LIBS
    "${ONNXRUNTIME_LIB_DIR}/*.dll"
    "${ONNXRUNTIME_LIB_DIR}/*.so*"
    "${ONNXRUNTIME_LIB_DIR}/*.dylib"
)

if(WIN32)
    set(ONNXRUNTIME_LINK_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
elseif(APPLE)
    set(ONNXRUNTIME_LINK_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
else()
    set(ONNXRUNTIME_LINK_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
endif()

if(NOT EXISTS ${ONNXRUNTIME_LINK_LIB})
    message(FATAL_ERROR "Failed to locate ONNX Runtime library at ${ONNXRUNTIME_LINK_LIB}")
endif()

# OpenCV headers (we only need the includes)
set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_ARCH "")

if(WIN32)
    set(OPENCV_ARCH "windowsx86-64")
elseif(APPLE)
    set(OPENCV_ARCH "osx-universal")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENCV_ARCH "linux-aarch64")
else()
    set(OPENCV_ARCH "linuxx86-64")
endif()

FetchContent_Declare(
    opencv_headers
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(opencv_headers)
if(NOT opencv_headers_POPULATED)
    FetchContent_Populate(opencv_headers)
endif()

FetchContent_Declare(
    opencv_libs
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(opencv_libs)
if(NOT opencv_libs_POPULATED)
    FetchContent_Populate(opencv_libs)
endif()

message(STATUS "OpenCV libs extracted to: ${opencv_libs_SOURCE_DIR}")

include_directories(${opencv_headers_SOURCE_DIR})

file(GLOB_RECURSE OPENCV_LIBRARY_FILES
    "${opencv_libs_SOURCE_DIR}/*.lib"
    "${opencv_libs_SOURCE_DIR}/*.a"
    "${opencv_libs_SOURCE_DIR}/*.so*"
)

# We need both the Java bindings AND the C++ component libraries for JNI to work
# Don't filter - use all libraries
message(STATUS "OpenCV library files: ${OPENCV_LIBRARY_FILES}")

file(GLOB_RECURSE OPENCV_RUNTIME_FILES
    "${opencv_libs_SOURCE_DIR}/*.dll"
    "${opencv_libs_SOURCE_DIR}/*.so*"
    "${opencv_libs_SOURCE_DIR}/*.dylib"
)

find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} SHARED src/main/native/cpp/onnx_jni.cpp)

if(WIN32 AND USE_DIRECTML)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DIRECTML)
endif()

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${opencv_headers_SOURCE_DIR}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${ONNXRUNTIME_LINK_LIB}
    ${OPENCV_LIBRARY_FILES}
        Threads::Threads
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\\$ORIGIN")

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
install(FILES ${ONNXRUNTIME_SHARED_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(OPENCV_RUNTIME_FILES)
    install(FILES ${OPENCV_RUNTIME_FILES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
