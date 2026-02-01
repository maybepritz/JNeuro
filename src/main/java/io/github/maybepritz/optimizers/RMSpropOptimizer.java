package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;
import java.util.HashMap;
import java.util.Map;

public class RMSpropOptimizer implements Optimizer {
    private final double learningRate;
    private final double decay;      // обычно 0.9 или 0.99
    private final double epsilon;    // обычно 1e-8

    // Хранилище накопленных квадратов градиентов для каждого слоя
    private final Map<String, Matrix> cache = new HashMap<>();

    public RMSpropOptimizer(double learningRate, double decay, double epsilon) {
        this.learningRate = learningRate;
        this.decay = decay;
        this.epsilon = epsilon;
    }

    public RMSpropOptimizer(double learningRate, double decay) {
        this(learningRate, decay, 1e-8);
    }

    public RMSpropOptimizer(double learningRate) {
        this(learningRate, 0.9, 1e-8);
    }

    @Override
    public Matrix update(Matrix weights, Matrix gradient, String layerId) {
        // Инициализация кэша если нужно
        if (!cache.containsKey(layerId)) {
            cache.put(layerId, weights.copy().scale(0)); // нули той же размерности
        }

        Matrix cacheCurrent = cache.get(layerId);

        // Обновляем кэш: cache = decay * cache + (1 - decay) * gradient²
        Matrix gradientSquared = gradient.elementMultiply(gradient);
        Matrix cacheNew = cacheCurrent.scale(decay)
                .add(gradientSquared.scale(1 - decay));

        // Обновление весов: weights = weights - lr * gradient / (sqrt(cache) + epsilon)
        Matrix sqrtCache = cacheNew.map(x -> Math.sqrt(x) + epsilon);
        Matrix normalizedGradient = gradient.elementMultiply(
                sqrtCache.map(x -> 1.0 / x)
        );
        Matrix newWeights = weights.subtract(normalizedGradient.scale(learningRate));

        // Сохраняем новый кэш
        cache.put(layerId, cacheNew);

        return newWeights;
    }

    @Override
    public void reset() {
        cache.clear();
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getDecay() {
        return decay;
    }

    public double getEpsilon() {
        return epsilon;
    }
}