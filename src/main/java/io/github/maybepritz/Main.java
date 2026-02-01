package io.github.maybepritz;

import io.github.maybepritz.activations.Sigmoid;
import io.github.maybepritz.activations.Tanh;
import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.config.TrainingConfig;
import io.github.maybepritz.core.NeuralNetwork;
import io.github.maybepritz.layers.DenseLayer;
import io.github.maybepritz.training.Trainer;

import java.util.Random;

public class Main {

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("   Аппроксимация функции f(x) = sin(x)");
        System.out.println("==========================================\n");

        // ========= 1. Генерация данных =========
        int samples = 1000;
        double[][] X = new double[samples][1];
        double[][] Y = new double[samples][1];

        Random rand = new Random(42);
        for (int i = 0; i < samples; i++) {
            // Равномерное распределение точек (не случайное!)
            double x = -Math.PI + (2 * Math.PI * i / samples);

            // Нормализация входа в [-1, 1]
            X[i][0] = x / Math.PI;

            // sin уже в [-1, 1], нормализуем в [0, 1] для Sigmoid выхода
            Y[i][0] = (Math.sin(x) + 1.0) / 2.0;
        }

        // Перемешиваем
        shuffleData(X, Y, rand);

        System.out.printf("Сгенерировано %d точек в диапазоне [-π, π]%n", samples);
        System.out.printf("Вход X: [%.2f, %.2f], Выход Y: [%.2f, %.2f]%n%n",
                -1.0, 1.0, 0.0, 1.0);

        // ========= 2. Конфигурация сети =========
        NetworkConfig netConfig = new NetworkConfig();
        netConfig.setWeightInit(NetworkConfig.WeightInit.XAVIER); // Xavier лучше для Tanh
        netConfig.useAdam(0.005);  // Уменьшил learning rate
        netConfig.setGradientClip(1.0);  // Клиппинг градиентов

        // Архитектура: 1 → 64 → 64 → 1
        // Используем Tanh в скрытых слоях (лучше для отрицательных значений)
        NeuralNetwork nn = new NeuralNetwork(netConfig);
        nn.addLayer(new DenseLayer(new LayerConfig(1, 64, new Tanh()), netConfig));
        nn.addLayer(new DenseLayer(new LayerConfig(64, 64, new Tanh()), netConfig));
        nn.addLayer(new DenseLayer(new LayerConfig(64, 1, new Sigmoid()), netConfig)); // [0,1] выход

        System.out.println("Архитектура: 1 → 64 (Tanh) → 64 (Tanh) → 1 (Sigmoid)");

        // ========= 3. Обучение =========
        TrainingConfig trainConfig = new TrainingConfig();
        trainConfig.setEpochs(500);
        trainConfig.setBatchSize(32);
        trainConfig.setLogInterval(50);
        trainConfig.setVerbose(true);
        trainConfig.setValidationSplit(0.1);

        // ⚠️ ВАЖНО: отключаем многопоточность для отладки!
        trainConfig.setNumThreads(1);

        Trainer trainer = new Trainer(nn, trainConfig);
        long start = System.currentTimeMillis();
        trainer.fit(X, Y);
        System.out.printf("\nОбучение: %.1f сек%n", (System.currentTimeMillis() - start) / 1000.0);

        // ========= 4. Тестирование =========
        System.out.println("\n=== Сравнение: реальный sin(x) vs предсказание ===");
        System.out.println("    x     | sin(x)  | predict | error");
        System.out.println("----------|---------|---------|-------");

        double totalError = 0;
        int testPoints = 9;
        for (int i = 0; i < testPoints; i++) {
            double x = -Math.PI + (2 * Math.PI * i / (testPoints - 1));
            double inputNorm = x / Math.PI;  // [-1, 1]
            double expected = Math.sin(x);   // [-1, 1]

            double[] output = nn.predict(new double[]{inputNorm});
            double predicted = output[0] * 2.0 - 1.0;  // [0,1] → [-1,1]

            double error = Math.abs(expected - predicted);
            totalError += error;

            System.out.printf(" %+6.3f  | %+6.3f  | %+6.3f  | %.4f%n",
                    x, expected, predicted, error);
        }
        System.out.printf("\nСредняя ошибка: %.4f%n", totalError / testPoints);

        // ========= 5. ASCII график =========
        System.out.println("\n=== ASCII График ===");
        printGraph(nn);
    }

    private static void shuffleData(double[][] X, double[][] Y, Random rand) {
        for (int i = X.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] tempX = X[i]; X[i] = X[j]; X[j] = tempX;
            double[] tempY = Y[i]; Y[i] = Y[j]; Y[j] = tempY;
        }
    }

    private static void printGraph(NeuralNetwork nn) {
        int width = 60;
        int height = 20;
        char[][] graph = new char[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                graph[i][j] = ' ';
            }
        }

        // Оси
        for (int j = 0; j < width; j++) graph[height / 2][j] = '─';
        for (int i = 0; i < height; i++) graph[i][width / 2] = '│';
        graph[height / 2][width / 2] = '┼';

        // Реальный sin (точки)
        for (int j = 0; j < width; j++) {
            double x = (j - width / 2.0) / (width / 2.0) * Math.PI;
            double y = Math.sin(x);
            int row = (int) ((1 - y) / 2 * (height - 1));
            if (row >= 0 && row < height) {
                graph[row][j] = '·';
            }
        }

        // Предсказание (звёзды)
        for (int j = 0; j < width; j++) {
            double x = (j - width / 2.0) / (width / 2.0) * Math.PI;
            double inputNorm = x / Math.PI;
            double[] output = nn.predict(new double[]{inputNorm});
            double y = output[0] * 2.0 - 1.0;
            int row = (int) ((1 - y) / 2 * (height - 1));
            if (row >= 0 && row < height) {
                graph[row][j] = (graph[row][j] == '·') ? '◉' : '*';
            }
        }

        System.out.println("  · = sin(x),  * = prediction,  ◉ = match");
        for (int i = 0; i < height; i++) {
            System.out.println(new String(graph[i]));
        }
    }
}