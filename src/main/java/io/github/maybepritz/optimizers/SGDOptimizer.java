package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;

public class SGDOptimizer implements Optimizer {
    private final double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Matrix update(Matrix weights, Matrix gradient, String layerId) {
        // weights = weights - lr * gradient
        return weights.subtract(gradient.scale(learningRate));
    }

    @Override
    public void reset() {
        // Нет состояния
    }
}