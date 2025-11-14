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

#include <jni.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace {

struct BoxRect {
    int left;
    int top;
    int right;
    int bottom;
};

struct DetectionCandidate {
    int classId;
    float score;
    BoxRect box;
};

struct OnnxDetector {
    std::unique_ptr<Ort::Session> session;
    std::vector<int64_t> inputShapeTemplate;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    bool channelsFirst{true};
    float lastMaxScore{std::numeric_limits<float>::lowest()};
};

jclass g_detectionResultClass = nullptr;
jclass g_runtimeExceptionClass = nullptr;
std::once_flag g_envInitFlag;

std::basic_string<ORTCHAR_T> ToOrtPath(const char* path) {
#ifdef _WIN32
    if (path == nullptr) {
        return {};
    }

    int wideLength = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (wideLength <= 0) {
        throw std::runtime_error("Failed to convert model path to wide string");
    }

    std::wstring widePath(static_cast<size_t>(wideLength), L'\0');
    int converted =
            MultiByteToWideChar(CP_UTF8, 0, path, -1, widePath.data(), wideLength);
    if (converted <= 0) {
        throw std::runtime_error("Failed to convert model path to wide string");
    }

    // MultiByteToWideChar writes the null terminator; trim it for std::wstring
    if (!widePath.empty() && widePath.back() == L'\0') {
        widePath.pop_back();
    }

    return widePath;
#else
    return std::basic_string<ORTCHAR_T>(path ? path : "");
#endif
}

Ort::Env& GetOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PhotonVisionOnnx");
    return env;
}

void ThrowRuntimeException(JNIEnv* env, const char* message) {
    if (g_runtimeExceptionClass != nullptr) {
        env->ThrowNew(g_runtimeExceptionClass, message);
    }
}

jobject CreateJavaDetection(JNIEnv* env, const DetectionCandidate& candidate) {
    if (g_detectionResultClass == nullptr) {
        ThrowRuntimeException(env, "Detection result class not loaded");
        return nullptr;
    }

    jmethodID ctor = env->GetMethodID(g_detectionResultClass, "<init>", "(IIIIFI)V");
    if (ctor == nullptr) {
        ThrowRuntimeException(env, "Failed to locate OnnxResult constructor");
        return nullptr;
    }

    return env->NewObject(
            g_detectionResultClass,
            ctor,
            candidate.box.left,
            candidate.box.top,
            candidate.box.right,
            candidate.box.bottom,
            candidate.score,
            candidate.classId);
}

float IoU(const BoxRect& a, const BoxRect& b) {
    const int x1 = std::max(a.left, b.left);
    const int y1 = std::max(a.top, b.top);
    const int x2 = std::min(a.right, b.right);
    const int y2 = std::min(a.bottom, b.bottom);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    const int intersection = (x2 - x1) * (y2 - y1);
    const int areaA = (a.right - a.left) * (a.bottom - a.top);
    const int areaB = (b.right - b.left) * (b.bottom - b.top);

    return static_cast<float>(intersection) /
           static_cast<float>(areaA + areaB - intersection);
}

std::vector<DetectionCandidate> NonMaxSuppression(
        std::vector<DetectionCandidate>& candidates, float threshold) {
    if (candidates.empty()) {
        return {};
    }

    std::sort(
            candidates.begin(),
            candidates.end(),
            [](const DetectionCandidate& lhs, const DetectionCandidate& rhs) {
                return lhs.score > rhs.score;
            });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<DetectionCandidate> results;
    results.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        const auto& current = candidates[i];
        results.push_back(current);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (candidates[j].classId != current.classId) {
                continue;
            }

            if (IoU(current.box, candidates[j].box) > threshold) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

DetectionCandidate BuildCandidate(
        float xCenter,
        float yCenter,
        float width,
        float height,
        float score,
        int classId,
        int imageWidth,
        int imageHeight) {
    const bool normalized = width <= 2.0f && height <= 2.0f;

    if (normalized) {
        xCenter *= static_cast<float>(imageWidth);
        yCenter *= static_cast<float>(imageHeight);
        width *= static_cast<float>(imageWidth);
        height *= static_cast<float>(imageHeight);
    }

    const float x1 = xCenter - width / 2.0f;
    const float y1 = yCenter - height / 2.0f;
    const float x2 = xCenter + width / 2.0f;
    const float y2 = yCenter + height / 2.0f;

    DetectionCandidate candidate{};
    candidate.classId = classId;
    candidate.score = score;
    candidate.box.left = static_cast<int>(std::round(std::clamp(x1, 0.0f, static_cast<float>(imageWidth))));
    candidate.box.top = static_cast<int>(std::round(std::clamp(y1, 0.0f, static_cast<float>(imageHeight))));
    candidate.box.right = static_cast<int>(std::round(std::clamp(x2, 0.0f, static_cast<float>(imageWidth))));
    candidate.box.bottom = static_cast<int>(std::round(std::clamp(y2, 0.0f, static_cast<float>(imageHeight))));
    return candidate;
}

void ParseSingleOutput(
        const Ort::Value& output,
        int imageWidth,
        int imageHeight,
        double boxThresh,
    int expectedClassCount,
    std::vector<DetectionCandidate>& outCandidates,
    OnnxDetector* detector) {
    const auto tensorInfo = output.GetTensorTypeAndShapeInfo();
    const auto dims = tensorInfo.GetShape();
    const float* data = output.GetTensorData<float>();

    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected output tensor rank for ONNX detector");
    }

    const int64_t dim1 = dims[dims.size() - 2];
    const int64_t dim2 = dims[dims.size() - 1];

    bool channelsFirst = dim1 < dim2;
    size_t channels = static_cast<size_t>(channelsFirst ? dim1 : dim2);
    size_t boxes = static_cast<size_t>(channelsFirst ? dim2 : dim1);

    if (channels < 4 || boxes == 0) {
        throw std::runtime_error("Malformed detection tensor");
    }

    size_t classOffset = 4;
    size_t classCount = channels > classOffset ? (channels - classOffset) : 0;
    bool hasObjectness = false;

    if (expectedClassCount > 0) {
        const size_t expected = static_cast<size_t>(expectedClassCount);

        if (channels == expected + 4) {
            classOffset = 4;
            classCount = expected;
            hasObjectness = false;
        } else if (channels == expected + 5) {
            classOffset = 5;
            classCount = expected;
            hasObjectness = true;
        } else if (channels >= expected + 4) {
            classOffset = channels - expected;
            if (classOffset < 4) {
                classOffset = 4;
            }
            classCount = std::min(expected, channels > classOffset ? channels - classOffset : 0);
            hasObjectness = classOffset > 4;
        } else {
            classOffset = channels > expected ? channels - expected : 4;
            if (classOffset < 4) {
                classOffset = 4;
            }
            classCount = channels > classOffset ? channels - classOffset : 0;
            hasObjectness = classOffset > 4;
        }
    } else {
        if (channels > 5) {
            classOffset = 5;
            classCount = channels - classOffset;
            hasObjectness = true;
        } else {
            classOffset = 4;
            classCount = channels > classOffset ? channels - classOffset : 0;
            hasObjectness = false;
        }
    }

    for (size_t i = 0; i < boxes; ++i) {
        auto valueAt = [&](size_t attribute, size_t idx) -> float {
            if (channelsFirst) {
                return data[attribute * boxes + idx];
            }
            return data[idx * channels + attribute];
        };

        const float xCenter = valueAt(0, i);
        const float yCenter = valueAt(1, i);
        const float width = valueAt(2, i);
        const float height = valueAt(3, i);

        const float objectness = hasObjectness ? valueAt(classOffset - 1, i) : 1.0f;

        if (classCount == 0) {
            const float score = objectness;
            if (score < boxThresh) {
                if (detector != nullptr && score > detector->lastMaxScore) {
                    detector->lastMaxScore = score;
                }
                continue;
            }
            outCandidates.push_back(
                    BuildCandidate(xCenter, yCenter, width, height, score, 0, imageWidth, imageHeight));
            continue;
        }

        float bestScore = 0.0f;
        int bestClass = -1;
        for (size_t cls = 0; cls < classCount; ++cls) {
            const size_t attributeIndex = classOffset + cls;
            if (attributeIndex >= channels) {
                break;
            }

            const float clsScore = valueAt(attributeIndex, i);
            const float score = objectness * clsScore;
            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(cls);
            }
        }

        if (detector != nullptr && bestScore > detector->lastMaxScore) {
            detector->lastMaxScore = bestScore;
        }

        if (bestClass < 0 || bestScore < boxThresh) {
            continue;
        }

        outCandidates.push_back(
                BuildCandidate(
                        xCenter, yCenter, width, height, bestScore, bestClass, imageWidth, imageHeight));
    }
}

void ParseSplitOutputs(
        const Ort::Value& boxesTensor,
        const Ort::Value& scoresTensor,
        int imageWidth,
        int imageHeight,
        double boxThresh,
    int expectedClassCount,
    std::vector<DetectionCandidate>& outCandidates,
    OnnxDetector* detector) {
    const auto boxesInfo = boxesTensor.GetTensorTypeAndShapeInfo();
    const auto scoresInfo = scoresTensor.GetTensorTypeAndShapeInfo();

    const auto boxesDims = boxesInfo.GetShape();
    const auto scoresDims = scoresInfo.GetShape();

    if (boxesDims.size() < 3 || scoresDims.size() < 3) {
        throw std::runtime_error("Unexpected split tensor dimensions");
    }

    const size_t boxesCount = static_cast<size_t>(boxesDims[boxesDims.size() - 2]);
    const size_t boxComponents = static_cast<size_t>(boxesDims[boxesDims.size() - 1]);
    size_t classCount = static_cast<size_t>(scoresDims[scoresDims.size() - 1]);
    if (expectedClassCount > 0) {
        classCount = std::min(classCount, static_cast<size_t>(expectedClassCount));
    }

    if (boxComponents < 4 || boxesCount == 0 || classCount == 0) {
        throw std::runtime_error("Malformed split outputs for detections");
    }

    const float* boxesData = boxesTensor.GetTensorData<float>();
    const float* scoresData = scoresTensor.GetTensorData<float>();

    for (size_t idx = 0; idx < boxesCount; ++idx) {
        const float* boxBase = boxesData + idx * boxComponents;
        const float* scoreBase = scoresData + idx * classCount;

        float bestScore = 0.0f;
        int bestClass = -1;
        for (size_t cls = 0; cls < classCount; ++cls) {
            const float score = scoreBase[cls];
            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(cls);
            }
        }

        if (detector != nullptr && bestScore > detector->lastMaxScore) {
            detector->lastMaxScore = bestScore;
        }

        if (bestClass < 0 || bestScore < boxThresh) {
            continue;
        }

        outCandidates.push_back(
                BuildCandidate(
                        boxBase[0],
                        boxBase[1],
                        boxBase[2],
                        boxBase[3],
                        bestScore,
                        bestClass,
                        imageWidth,
                        imageHeight));
    }
}

std::vector<DetectionCandidate> RunPostProcessing(
        std::vector<Ort::Value>& outputs,
        int imageWidth,
        int imageHeight,
        double boxThresh,
        double nmsThresh,
        int expectedClassCount,
        OnnxDetector* detector) {
    std::vector<DetectionCandidate> candidates;

    try {
        if (outputs.size() >= 2) {
            ParseSplitOutputs(
                    outputs[0],
                    outputs[1],
                    imageWidth,
                    imageHeight,
                    boxThresh,
                    expectedClassCount,
                    candidates,
                    detector);
        } else if (!outputs.empty()) {
            ParseSingleOutput(
                    outputs[0],
                    imageWidth,
                    imageHeight,
                    boxThresh,
                    expectedClassCount,
                    candidates,
                    detector);
        }
    } catch (const std::exception& ex) {
        throw;
    }

    return NonMaxSuppression(candidates, static_cast<float>(nmsThresh));
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK) {
        return JNI_ERR;
    }

    std::call_once(g_envInitFlag, [env]() {
        jclass runtimeClass = env->FindClass("java/lang/RuntimeException");
        if (runtimeClass != nullptr) {
            g_runtimeExceptionClass = static_cast<jclass>(env->NewGlobalRef(runtimeClass));
            env->DeleteLocalRef(runtimeClass);
        }

        jclass detectionClass = env->FindClass("org/photonvision/onnx/OnnxJNI$OnnxResult");
        if (detectionClass != nullptr) {
            g_detectionResultClass = static_cast<jclass>(env->NewGlobalRef(detectionClass));
            env->DeleteLocalRef(detectionClass);
        }
    });

    if (g_detectionResultClass == nullptr) {
        return JNI_ERR;
    }

    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) == JNI_OK) {
        if (g_detectionResultClass != nullptr) {
            env->DeleteGlobalRef(g_detectionResultClass);
            g_detectionResultClass = nullptr;
        }
        if (g_runtimeExceptionClass != nullptr) {
            env->DeleteGlobalRef(g_runtimeExceptionClass);
            g_runtimeExceptionClass = nullptr;
        }
    }
}

JNIEXPORT jlong JNICALL Java_org_photonvision_onnx_OnnxJNI_create(
        JNIEnv* env, jclass, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) {
        ThrowRuntimeException(env, "Failed to obtain model path");
        return 0;
    }

    std::unique_ptr<OnnxDetector> detector;
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        options.DisableMemPattern();
        options.SetLogSeverityLevel(3);

    detector = std::make_unique<OnnxDetector>();
    auto ortPath = ToOrtPath(path);
    detector->session = std::make_unique<Ort::Session>(GetOrtEnv(), ortPath.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t inputCount = detector->session->GetInputCount();
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = detector->session->GetInputNameAllocated(i, allocator);
            detector->inputNames.emplace_back(name.get());
        }

        const size_t outputCount = detector->session->GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = detector->session->GetOutputNameAllocated(i, allocator);
            detector->outputNames.emplace_back(name.get());
        }

        auto tensorInfo =
                detector->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        detector->inputShapeTemplate = tensorInfo.GetShape();

        if (detector->inputShapeTemplate.size() >= 4) {
            const int64_t dim1 = detector->inputShapeTemplate[1];
            const int64_t dimLast = detector->inputShapeTemplate[3];
            if (dim1 == 3 || dim1 == -1) {
                detector->channelsFirst = true;
            } else if (dimLast == 3 || dimLast == -1) {
                detector->channelsFirst = false;
            } else {
                detector->channelsFirst = dim1 <= dimLast;
            }
        }
    } catch (const Ort::Exception& ex) {
        env->ReleaseStringUTFChars(modelPath, path);
        ThrowRuntimeException(env, ex.what());
        return 0;
    } catch (const std::exception& ex) {
        env->ReleaseStringUTFChars(modelPath, path);
        ThrowRuntimeException(env, ex.what());
        return 0;
    }

    env->ReleaseStringUTFChars(modelPath, path);
    return reinterpret_cast<jlong>(detector.release());
}

JNIEXPORT void JNICALL Java_org_photonvision_onnx_OnnxJNI_destroy(
        JNIEnv*, jclass, jlong handle) {
    auto* detector = reinterpret_cast<OnnxDetector*>(handle);
    delete detector;
}

JNIEXPORT jobjectArray JNICALL Java_org_photonvision_onnx_OnnxJNI_detect(
        JNIEnv* env,
        jclass,
        jlong handle,
        jlong matPtr,
        jdouble boxThresh,
    jdouble nmsThresh,
    jint expectedClassCount) {
    auto* detector = reinterpret_cast<OnnxDetector*>(handle);
    if (detector == nullptr || detector->session == nullptr) {
        ThrowRuntimeException(env, "Invalid ONNX detector instance");
        return nullptr;
    }

    cv::Mat* mat = reinterpret_cast<cv::Mat*>(matPtr);
    if (mat == nullptr || mat->empty()) {
        ThrowRuntimeException(env, "Input frame was null or empty");
        return nullptr;
    }

    if (mat->channels() != 3) {
        ThrowRuntimeException(env, "ONNX detector expects 3-channel images");
        return nullptr;
    }

    const int imageWidth = mat->cols;
    const int imageHeight = mat->rows;
    const size_t channels = static_cast<size_t>(mat->channels());

    auto shape = detector->inputShapeTemplate;
    if (shape.size() >= 4) {
        if (detector->channelsFirst) {
            shape[0] = 1;
            shape[1] = static_cast<int64_t>(channels);
            shape[2] = imageHeight;
            shape[3] = imageWidth;
        } else {
            shape[0] = 1;
            shape[1] = imageHeight;
            shape[2] = imageWidth;
            shape[3] = static_cast<int64_t>(channels);
        }
    } else if (shape.size() == 3) {
        shape[0] = 1;
        shape[1] = imageHeight;
        shape[2] = imageWidth;
    } else {
        ThrowRuntimeException(env, "Unsupported input tensor rank for ONNX detector");
        return nullptr;
    }

    const size_t tensorSize = static_cast<size_t>(imageWidth) * imageHeight * channels;
    std::vector<float> inputTensor(tensorSize, 0.0f);

    if (detector->channelsFirst) {
        const size_t planeSize = static_cast<size_t>(imageWidth) * imageHeight;
        for (int y = 0; y < imageHeight; ++y) {
            const auto* row = mat->ptr<cv::Vec3b>(y);
            for (int x = 0; x < imageWidth; ++x) {
                const cv::Vec3b& pixel = row[x];
                const float r = static_cast<float>(pixel[2]) / 255.0f;
                const float g = static_cast<float>(pixel[1]) / 255.0f;
                const float b = static_cast<float>(pixel[0]) / 255.0f;
                const size_t index = static_cast<size_t>(y) * imageWidth + x;
                inputTensor[index] = r;
                inputTensor[planeSize + index] = g;
                inputTensor[2 * planeSize + index] = b;
            }
        }
    } else {
        for (int y = 0; y < imageHeight; ++y) {
            const auto* row = mat->ptr<cv::Vec3b>(y);
            for (int x = 0; x < imageWidth; ++x) {
                const cv::Vec3b& pixel = row[x];
                const float r = static_cast<float>(pixel[2]) / 255.0f;
                const float g = static_cast<float>(pixel[1]) / 255.0f;
                const float b = static_cast<float>(pixel[0]) / 255.0f;
                const size_t base = (static_cast<size_t>(y) * imageWidth + x) * channels;
                inputTensor[base] = r;
                inputTensor[base + 1] = g;
                inputTensor[base + 2] = b;
            }
        }
    }

    std::vector<const char*> inputNames;
    inputNames.reserve(detector->inputNames.size());
    for (const auto& name : detector->inputNames) {
        inputNames.push_back(name.c_str());
    }

    std::vector<const char*> outputNames;
    outputNames.reserve(detector->outputNames.size());
    for (const auto& name : detector->outputNames) {
        outputNames.push_back(name.c_str());
    }

    std::vector<Ort::Value> outputs;
    try {
        auto inputTensorValue = Ort::Value::CreateTensor<float>(
                detector->memoryInfo,
                inputTensor.data(),
                inputTensor.size(),
                shape.data(),
                shape.size());

        outputs = detector->session->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                &inputTensorValue,
                inputNames.size(),
                outputNames.data(),
                outputNames.size());
    } catch (const Ort::Exception& ex) {
        ThrowRuntimeException(env, ex.what());
        return nullptr;
    } catch (const std::exception& ex) {
        ThrowRuntimeException(env, ex.what());
        return nullptr;
    }

    detector->lastMaxScore = std::numeric_limits<float>::lowest();

    std::vector<DetectionCandidate> detections;
    try {
    detections = RunPostProcessing(
        outputs,
        imageWidth,
        imageHeight,
        boxThresh,
        nmsThresh,
                static_cast<int>(expectedClassCount),
                detector);
    } catch (const std::exception& ex) {
        ThrowRuntimeException(env, ex.what());
        return nullptr;
    }

    jobjectArray resultArray = env->NewObjectArray(
            detections.size(),
            g_detectionResultClass,
            nullptr);

    for (size_t i = 0; i < detections.size(); ++i) {
        jobject detection = CreateJavaDetection(env, detections[i]);
        env->SetObjectArrayElement(resultArray, static_cast<jsize>(i), detection);
        env->DeleteLocalRef(detection);
    }

    return resultArray;
}

JNIEXPORT jboolean JNICALL Java_org_photonvision_onnx_OnnxJNI_isQuantized(
        JNIEnv*, jclass, jlong) {
    return JNI_FALSE;
}

JNIEXPORT jdouble JNICALL Java_org_photonvision_onnx_OnnxJNI_getLastMaxScore(
        JNIEnv* env, jclass, jlong handle) {
    auto* detector = reinterpret_cast<OnnxDetector*>(handle);
    if (detector == nullptr) {
        ThrowRuntimeException(env, "Invalid ONNX detector instance");
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (detector->lastMaxScore == std::numeric_limits<float>::lowest()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return static_cast<double>(detector->lastMaxScore);
}

JNIEXPORT jintArray JNICALL Java_org_photonvision_onnx_OnnxJNI_getInputSize(
        JNIEnv* env, jclass, jlong handle) {
    auto* detector = reinterpret_cast<OnnxDetector*>(handle);
    if (detector == nullptr) {
        ThrowRuntimeException(env, "Invalid ONNX detector instance");
        return nullptr;
    }

    int width = -1;
    int height = -1;

    const auto& shape = detector->inputShapeTemplate;
    if (shape.size() >= 4) {
        if (detector->channelsFirst) {
            if (shape[2] > 0) {
                height = static_cast<int>(shape[2]);
            }
            if (shape[3] > 0) {
                width = static_cast<int>(shape[3]);
            }
        } else {
            if (shape[1] > 0) {
                height = static_cast<int>(shape[1]);
            }
            if (shape[2] > 0) {
                width = static_cast<int>(shape[2]);
            }
        }
    } else if (shape.size() == 3) {
        if (shape[shape.size() - 2] > 0) {
            height = static_cast<int>(shape[shape.size() - 2]);
        }
        if (shape[shape.size() - 1] > 0) {
            width = static_cast<int>(shape[shape.size() - 1]);
        }
    }

    jintArray result = env->NewIntArray(2);
    if (result == nullptr) {
        ThrowRuntimeException(env, "Failed to allocate input size array");
        return nullptr;
    }

    jint buffer[2] = {width, height};
    env->SetIntArrayRegion(result, 0, 2, buffer);
    return result;
}

}  // extern "C"
