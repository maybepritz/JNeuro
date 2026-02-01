package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;
import io.github.maybepritz.utils.MatrixFactory;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class AdamOptimizer implements Optimizer {
    private final double learningRate;
    private final double beta1;  // обычно 0.9
    private final double beta2;  // обычно 0.999
    private final double epsilon; // обычно 1e-8

    // Состояние для каждого слоя
    private final Map<String, Matrix> m = new ConcurrentHashMap<>(); // момент 1-го порядка
    private final Map<String, Matrix> v = new ConcurrentHashMap<>(); // момент 2-го порядка
    private final Map<String, Integer> t = new ConcurrentHashMap<>(); // счётчик обновлений

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
            m.put(layerId, MatrixFactory.create(weights.getRows(), weights.getCols()));
            v.put(layerId, MatrixFactory.create(weights.getRows(), weights.getCols()));
            t.put(layerId, 0);
        }

        int timestep = t.get(layerId) + 1;
        t.put(layerId, timestep);

        Matrix mCurrent = m.get(layerId);
        Matrix vCurrent = v.get(layerId);

        // ✅ Используем copy() чтобы не мутировать состояние и градиент
        Matrix mNew = mCurrent.copy().scale(beta1)
                .add(gradient.copy().scale(1 - beta1));

        Matrix gradientSquared = gradient.elementMultiply(gradient);
        Matrix vNew = vCurrent.copy().scale(beta2)
                .add(gradientSquared.scale(1 - beta2));

        // Коррекция смещения
        double beta1Correction = 1.0 / (1.0 - Math.pow(beta1, timestep));
        double beta2Correction = 1.0 / (1.0 - Math.pow(beta2, timestep));

        Matrix mHat = mNew.copy().scale(beta1Correction);
        Matrix vHat = vNew.copy().scale(beta2Correction);

        // ✅ Более эффективно: объединить sqrt и деление
        Matrix update = mHat.elementMultiply(
                vHat.map(x -> 1.0 / (Math.sqrt(x) + epsilon))
        );

        Matrix newWeights = weights.copy().subtract(update.scale(learningRate));

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