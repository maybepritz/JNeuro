package io.github.maybepritz.training;

import io.github.maybepritz.config.TrainingConfig;
import io.github.maybepritz.core.NeuralNetwork;

import java.util.Random;

public class Trainer {
    private final NeuralNetwork network;
    private final TrainingConfig config;
    private final Random random = new Random();

    public Trainer(NeuralNetwork network, TrainingConfig config) {
        this.network = network;
        this.config = config;
    }

    public Trainer(NeuralNetwork network) {
        this(network, new TrainingConfig());
    }

    public void fit(double[][] X, double[][] Y) {
        // Разделение на train/validation
        double valSplit = config.getValidationSplit();
        int valSize = (int) (X.length * valSplit);
        int trainSize = X.length - valSize;

        double[][] trainX, trainY, valX = null, valY = null;

        if (valSize > 0) {
            // Перемешиваем индексы для случайного разбиения
            int[] allIndices = createShuffledIndices(X.length);

            trainX = new double[trainSize][];
            trainY = new double[trainSize][];
            valX = new double[valSize][];
            valY = new double[valSize][];

            for (int i = 0; i < trainSize; i++) {
                trainX[i] = X[allIndices[i]];
                trainY[i] = Y[allIndices[i]];
            }
            for (int i = 0; i < valSize; i++) {
                valX[i] = X[allIndices[trainSize + i]];
                valY[i] = Y[allIndices[trainSize + i]];
            }

            if (config.isVerbose()) {
                System.out.printf("Train: %d samples, Validation: %d samples%n", trainSize, valSize);
            }
        } else {
            trainX = X;
            trainY = Y;
        }

        // Индексы для обучающей выборки
        int[] indices = new int[trainX.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;

        double bestLoss = Double.MAX_VALUE;
        int patienceCounter = 0;

        if (config.isVerbose()) {
            System.out.println("Начало обучения...");
            System.out.println("==================");
        }

        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            // Перемешивание
            if (config.isShuffle()) {
                shuffle(indices);
            }

            // Обучение по батчам
            double trainLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < trainX.length; i += config.getBatchSize()) {
                int end = Math.min(i + config.getBatchSize(), trainX.length);

                for (int j = i; j < end; j++) {
                    int idx = indices[j];
                    network.train(trainX[idx], trainY[idx]);
                }

                // Loss после батча
                for (int j = i; j < end; j++) {
                    int idx = indices[j];
                    trainLoss += calculateLoss(network.forward(trainX[idx]), trainY[idx]);
                }
                batchCount += (end - i);
            }

            double avgTrainLoss = trainLoss / batchCount;

            // Validation loss
            double avgValLoss = 0;
            if (valX != null) {
                double valLoss = 0;
                for (int i = 0; i < valX.length; i++) {
                    valLoss += calculateLoss(network.predict(valX[i]), valY[i]);
                }
                avgValLoss = valLoss / valX.length;
            }

            // Логирование
            if (config.isVerbose() && epoch % config.getLogInterval() == 0) {
                if (valX != null) {
                    System.out.printf("Epoch %5d | Train Loss: %.6f | Val Loss: %.6f%n",
                            epoch, avgTrainLoss, avgValLoss);
                } else {
                    System.out.printf("Epoch %5d | Loss: %.6f%n", epoch, avgTrainLoss);
                }
            }

            // Early stopping (по validation loss если есть, иначе по train loss)
            double monitorLoss = (valX != null) ? avgValLoss : avgTrainLoss;

            if (config.isEarlyStopping()) {
                if (monitorLoss < bestLoss - config.getMinDelta()) {
                    bestLoss = monitorLoss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                    if (patienceCounter >= config.getPatience()) {
                        if (config.isVerbose()) {
                            System.out.printf("Early stopping at epoch %d (best loss: %.6f)%n",
                                    epoch, bestLoss);
                        }
                        break;
                    }
                }
            }
        }

        if (config.isVerbose()) {
            System.out.println("Обучение завершено!");
        }
    }

    /**
     * Оценка модели на тестовых данных
     */
    public double evaluate(double[][] X, double[][] y) {
        double totalLoss = 0;
        for (int i = 0; i < X.length; i++) {
            totalLoss += calculateLoss(network.predict(X[i]), y[i]);
        }
        return totalLoss / X.length;
    }

    /**
     * MSE Loss
     */
    private double calculateLoss(double[] predicted, double[] actual) {
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            double diff = actual[i] - predicted[i];
            sum += diff * diff;
        }
        return sum / actual.length;
    }

    private void shuffle(int[] arr) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    private int[] createShuffledIndices(int length) {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) indices[i] = i;
        shuffle(indices);
        return indices;
    }
}