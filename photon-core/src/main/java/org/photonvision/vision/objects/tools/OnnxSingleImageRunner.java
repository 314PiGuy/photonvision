package org.photonvision.vision.objects.tools;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkPropertyManager.ModelProperties;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.objects.OnnxModel;
import org.photonvision.vision.objects.OnnxObjectDetector;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/** Small CLI helper that runs the ONNX detector once against a single image. */
public final class OnnxSingleImageRunner {
    private OnnxSingleImageRunner() {}

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception ex) {
            System.err.println("Failed to execute ONNX detector: " + ex.getMessage());
            ex.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void run(String[] args) throws IOException {
        Map<String, String> options = parseOptions(args);

        Path modelPath = requirePath(options, "model");
        Path imagePath = requirePath(options, "image");

        LinkedList<String> labels = loadLabels(options);
        int explicitClassCount = parseInt(options.get("class-count"), labels.size());
        if (labels.isEmpty()) {
            if (explicitClassCount <= 0) {
                throw new IllegalArgumentException(
                        "Provide --labels or a positive --class-count for the model");
            }
            labels = new LinkedList<>();
            for (int i = 0; i < explicitClassCount; i++) {
                labels.add("class" + i);
            }
        } else if (explicitClassCount > 0 && explicitClassCount != labels.size()) {
            System.err.println(
                    "Warning: --class-count="
                            + explicitClassCount
                            + " differs from label count "
                            + labels.size()
                            + "; using labels size");
            explicitClassCount = labels.size();
        } else {
            explicitClassCount = labels.size();
        }

        int inputWidth = parseInt(options.get("width"), 640);
        int inputHeight = parseInt(options.get("height"), 640);
        double boxThreshold = parseDouble(options.get("conf"), 0.25);
        double nmsThreshold = parseDouble(options.get("nms"), 0.45);
        String nickname = options.getOrDefault("nickname", modelPath.getFileName().toString());

        System.setProperty("java.awt.headless", "true");
        if (!TestUtils.loadLibraries()) {
            throw new IllegalStateException("Failed to load native dependencies for OpenCV/WPILib");
        }

        ModelProperties properties =
                new ModelProperties(
                        modelPath,
                        nickname,
                        labels,
                        inputWidth,
                        inputHeight,
                        Family.ONNX,
                        Version.YOLOV8);

        OnnxObjectDetector detector = null;
        Mat image = new Mat();
        try {
            OnnxModel model = new OnnxModel(properties);
            detector = (OnnxObjectDetector) model.load();

            image = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_COLOR);
            if (image.empty()) {
                throw new IllegalArgumentException("Failed to read image " + imagePath);
            }

            List<NeuralNetworkPipeResult> detections =
                    detector.detect(image, nmsThreshold, boxThreshold);

            System.out.printf(Locale.US, "Total detections: %d%n", detections.size());
            for (int i = 0; i < detections.size(); i++) {
                NeuralNetworkPipeResult detection = detections.get(i);
                Rect2d bbox = detection.bbox();
                String label =
                        detection.classIdx() >= 0 && detection.classIdx() < explicitClassCount
                                ? labels.get(detection.classIdx())
                                : Integer.toString(detection.classIdx());
                System.out.printf(
                        Locale.US,
                        "#%d class=%s (idx=%d) conf=%.4f bbox=[x=%.1f y=%.1f w=%.1f h=%.1f]%n",
                        i,
                        label,
                        detection.classIdx(),
                        detection.confidence(),
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height);
            }

            double rawMaxScore = detector.getLastMaxScore();
            if (!Double.isNaN(rawMaxScore)) {
                System.out.printf(Locale.US, "Max raw class score observed: %.4f%n", rawMaxScore);
            }
        } finally {
            if (detector != null) {
                detector.release();
            }
            if (!image.empty()) {
                image.release();
            }
        }
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String token = args[i];
            if (!token.startsWith("--")) {
                throw new IllegalArgumentException(
                        "Unexpected argument '" + token + "'; use --key value form");
            }

            String key;
            String value;
            int equals = token.indexOf('=');
            if (equals > 2) {
                key = token.substring(2, equals);
                value = token.substring(equals + 1);
            } else {
                key = token.substring(2);
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("Missing value for --" + key);
                }
                value = args[++i];
            }

            options.put(key, value);
        }
        return options;
    }

    private static Path requirePath(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing required --" + key + " argument");
        }
        return Paths.get(value).toAbsolutePath().normalize();
    }

    private static LinkedList<String> loadLabels(Map<String, String> options) throws IOException {
        String labelsPath = options.get("labels");
        if (labelsPath == null || labelsPath.isBlank()) {
            return new LinkedList<>();
        }

        List<String> lines = Files.readAllLines(Paths.get(labelsPath));
        LinkedList<String> labels = new LinkedList<>();
        for (String line : lines) {
            String trimmed = line.trim();
            if (!trimmed.isEmpty() && !trimmed.startsWith("#")) {
                labels.add(trimmed);
            }
        }
        return labels;
    }

    private static int parseInt(String value, int defaultValue) {
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        return Integer.parseInt(value);
    }

    private static double parseDouble(String value, double defaultValue) {
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        return Double.parseDouble(value);
    }
}
