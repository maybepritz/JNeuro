package io.github.maybepritz;

import io.github.maybepritz.activations.ReLU;
import io.github.maybepritz.activations.Sigmoid;
import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.config.TrainingConfig;
import io.github.maybepritz.core.NeuralNetwork;
import io.github.maybepritz.layers.DenseLayer;
import io.github.maybepritz.layers.SoftmaxLayer;
import io.github.maybepritz.training.Trainer;
import io.github.maybepritz.utils.MatrixFactory;

import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class Main {

    private static final String MNIST_DIR = "mnist_data";
    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";
    private static final String BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/";

    public static void main(String[] args) throws Exception {
        System.out.println("===========================================");
        System.out.println("   MNIST - Распознавание рукописных цифр");
        System.out.println("===========================================\n");

        // ========= 1. Загрузка данных =========
        System.out.println("Загрузка MNIST...");
        downloadMnist();

        double[][] trainImages = loadImages(MNIST_DIR + "/" + TRAIN_IMAGES, 60000);
        int[] trainLabels = loadLabels(MNIST_DIR + "/" + TRAIN_LABELS, 60000);
        double[][] testImages = loadImages(MNIST_DIR + "/" + TEST_IMAGES, 10000);
        int[] testLabels = loadLabels(MNIST_DIR + "/" + TEST_LABELS, 10000);

        System.out.printf("Train: %d samples, Test: %d samples%n", trainImages.length, testImages.length);
        System.out.printf("Image size: %d pixels (28x28)%n%n", trainImages[0].length);

        // Конвертируем labels в one-hot encoding
        double[][] trainY = toOneHot(trainLabels, 10);
        double[][] testY = toOneHot(testLabels, 10);

        // ========= 2. Конфигурация сети =========
        NetworkConfig netConfig = new NetworkConfig();
        netConfig.setLearningRate(0.001);  // Уменьшил! Было 0.01
        netConfig.setDropoutRate(0.0);
        netConfig.setL2Regularization(0.0);  // Убрал регуляризацию
        netConfig.setWeightInit(NetworkConfig.WeightInit.XAVIER);  // XAVIER вместо HE
        netConfig.setGradientClip(1.0);  // Добавь клиппинг градиентов

        MatrixFactory.setBackend(MatrixFactory.Backend.CPU);

        // ========= 3. Архитектура сети =========
        // 784 входов (28x28 пикселей) → 128 → 64 → 10 выходов (цифры 0-9)
        NeuralNetwork nn = new NeuralNetwork(netConfig);
        nn.addLayer(new DenseLayer(new LayerConfig(784, 128, new ReLU()), netConfig));
        nn.addLayer(new DenseLayer(new LayerConfig(128, 64, new ReLU()), netConfig));
        nn.addLayer(new DenseLayer(new LayerConfig(64, 10, new Sigmoid()), netConfig));
        nn.addLayer(new SoftmaxLayer());

        System.out.println("Архитектура: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)");
        System.out.println();

        // ========= 4. Конфигурация обучения =========
        TrainingConfig trainConfig = new TrainingConfig();
        trainConfig.setEpochs(100);
        trainConfig.setBatchSize(32);
        trainConfig.setLogInterval(10);
        trainConfig.setShuffle(true);
        trainConfig.setVerbose(true);
        trainConfig.setEarlyStopping(false);

        // ========= 5. Обучение =========
        System.out.println("Начало обучения...");
        System.out.println("==================");

        long startTime = System.currentTimeMillis();

        // Используем подвыборку для ускорения (10000 из 60000)
        int trainSubset = 10000;
        double[][] trainXSubset = new double[trainSubset][];
        double[][] trainYSubset = new double[trainSubset][];

        Random rand = new Random(42);
        int[] indices = new int[60000];
        for (int i = 0; i < 60000; i++) indices[i] = i;
        shuffle(indices, rand);

        for (int i = 0; i < trainSubset; i++) {
            trainXSubset[i] = trainImages[indices[i]];
            trainYSubset[i] = trainY[indices[i]];
        }

        Trainer trainer = new Trainer(nn, trainConfig);
        trainer.fit(trainXSubset, trainYSubset);

        long endTime = System.currentTimeMillis();
        System.out.printf("\nОбучение завершено за %.1f сек%n", (endTime - startTime) / 1000.0);

        // ========= 6. Оценка на тестовых данных =========
        System.out.println("\n=== Оценка на тестовой выборке ===");

        int correct = 0;
        int testSubset = Math.min(1000, testImages.length);

        for (int i = 0; i < testSubset; i++) {
            double[] output = nn.predict(testImages[i]);
            int predicted = argmax(output);
            if (predicted == testLabels[i]) correct++;
        }

        double accuracy = (double) correct / testSubset * 100;
        System.out.printf("Точность: %d/%d (%.2f%%)%n", correct, testSubset, accuracy);

        // ========= 7. Примеры предсказаний =========
        System.out.println("\n=== Примеры предсказаний ===");

        for (int i = 0; i < 10; i++) {
            int idx = rand.nextInt(testImages.length);
            double[] output = nn.predict(testImages[idx]);
            int predicted = argmax(output);
            double confidence = output[predicted] * 100;

            System.out.printf("Изображение #%d: предсказано=%d, реально=%d, уверенность=%.1f%% %s%n",
                    idx, predicted, testLabels[idx], confidence,
                    predicted == testLabels[idx] ? "✓" : "✗");
        }

        // ========= 8. Визуализация одной цифры =========
        System.out.println("\n=== Визуализация цифры ===");
        int sampleIdx = 0;
        printDigit(testImages[sampleIdx]);
        double[] output = nn.predict(testImages[sampleIdx]);
        System.out.printf("Предсказание: %d (уверенность: %.1f%%)%n",
                argmax(output), output[argmax(output)] * 100);
        System.out.printf("Реальная цифра: %d%n", testLabels[sampleIdx]);

        // ========= 9. Сохранение модели =========
        try {
            nn.save("mnist_model.zip");
            System.out.println("\n✓ Модель сохранена в mnist_model.zip");
        } catch (Exception e) {
            System.err.println("Ошибка сохранения: " + e.getMessage());
        }
    }

    // ==================== ЗАГРУЗКА MNIST ====================

    private static void downloadMnist() throws IOException {
        Path dir = Paths.get(MNIST_DIR);
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
        }

        String[] files = {TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS};
        for (String file : files) {
            Path path = dir.resolve(file);
            if (!Files.exists(path)) {
                System.out.println("Скачивание " + file + "...");
                downloadFile(BASE_URL + file, path);
            }
        }
        System.out.println("MNIST загружен!\n");
    }

    private static void downloadFile(String urlStr, Path dest) throws IOException {
        URL url = new URL(urlStr);
        try (InputStream in = url.openStream()) {
            Files.copy(in, dest, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static double[][] loadImages(String path, int count) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new GZIPInputStream(new FileInputStream(path)))) {

            int magic = dis.readInt();
            int numImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            double[][] images = new double[Math.min(count, numImages)][rows * cols];

            for (int i = 0; i < images.length; i++) {
                for (int j = 0; j < rows * cols; j++) {
                    // Нормализация: 0-255 → 0-1
                    images[i][j] = (dis.readUnsignedByte()) / 255.0;
                }
            }
            return images;
        }
    }

    private static int[] loadLabels(String path, int count) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new GZIPInputStream(new FileInputStream(path)))) {

            int magic = dis.readInt();
            int numLabels = dis.readInt();

            int[] labels = new int[Math.min(count, numLabels)];
            for (int i = 0; i < labels.length; i++) {
                labels[i] = dis.readUnsignedByte();
            }
            return labels;
        }
    }

    // ==================== УТИЛИТЫ ====================

    private static double[][] toOneHot(int[] labels, int numClasses) {
        double[][] oneHot = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            oneHot[i][labels[i]] = 1.0;
        }
        return oneHot;
    }

    private static int argmax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static void shuffle(int[] arr, Random rand) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    private static void printDigit(double[] pixels) {
        System.out.println("┌" + "──".repeat(28) + "┐");
        for (int row = 0; row < 28; row++) {
            System.out.print("│");
            for (int col = 0; col < 28; col++) {
                double val = pixels[row * 28 + col];
                if (val > 0.8) System.out.print("██");
                else if (val > 0.5) System.out.print("▓▓");
                else if (val > 0.3) System.out.print("░░");
                else System.out.print("  ");
            }
            System.out.println("│");
        }
        System.out.println("└" + "──".repeat(28) + "┘");
    }
}