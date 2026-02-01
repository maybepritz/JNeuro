package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;
import java.util.HashMap;
import java.util.Map;

public class AdamOptimizer implements Optimizer {
    private final double learningRate;
    private final double beta1;  // обычно 0.9
    private final double beta2;  // обычно 0.999
    private final double epsilon; // обычно 1e-8

    // Состояние для каждого слоя
    private final Map<String, Matrix> m = new HashMap<>(); // момент 1-го порядка
    private final Map<String, Matrix> v = new HashMap<>(); // момент 2-го порядка
    private final Map<String, Integer> t = new HashMap<>(); // счётчик обновлений

    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public Matrix update(Matrix weights, Matrix gradient, String layerId) {
        // Инициализация состояния если нужно
        if (!m.containsKey(layerId)) {
            m.put(layerId, weights.copy().scale(0)); // нули той же размерности
            v.put(layerId, weights.copy().scale(0));
            t.put(layerId, 0);
        }

        // Увеличиваем счётчик
        int timestep = t.get(layerId) + 1;
        t.put(layerId, timestep);

        // Получаем текущее состояние
        Matrix mCurrent = m.get(layerId);
        Matrix vCurrent = v.get(layerId);

        // Обновляем моменты
        // m = beta1 * m + (1 - beta1) * gradient
        Matrix mNew = mCurrent.scale(beta1).add(gradient.scale(1 - beta1));

        // v = beta2 * v + (1 - beta2) * gradient²
        Matrix gradientSquared = gradient.elementMultiply(gradient);
        Matrix vNew = vCurrent.scale(beta2).add(gradientSquared.scale(1 - beta2));

        // Коррекция смещения
        double beta1Pow = Math.pow(beta1, timestep);
        double beta2Pow = Math.pow(beta2, timestep);

        Matrix mHat = mNew.scale(1.0 / (1.0 - beta1Pow));
        Matrix vHat = vNew.scale(1.0 / (1.0 - beta2Pow));

        // Обновление весов
        // weights = weights - lr * mHat / (sqrt(vHat) + epsilon)
        Matrix vHatSqrt = vHat.map(x -> Math.sqrt(x) + epsilon);
        Matrix update = mHat.elementMultiply(vHatSqrt.map(x -> 1.0 / x));
        Matrix newWeights = weights.subtract(update.scale(learningRate));

        // Сохраняем состояние
        m.put(layerId, mNew);
        v.put(layerId, vNew);

        return newWeights;
    }

    @Override
    public void reset() {
        m.clear();
        v.clear();
        t.clear();
    }
}