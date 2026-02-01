package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;
import java.util.HashMap;
import java.util.Map;

public class MomentumOptimizer implements Optimizer {
    private final double learningRate;
    private final double momentum; // обычно 0.9

    private final Map<String, Matrix> velocity = new HashMap<>();

    public MomentumOptimizer(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    public MomentumOptimizer(double learningRate) {
        this(learningRate, 0.9);
    }

    @Override
    public Matrix update(Matrix weights, Matrix gradient, String layerId) {
        if (!velocity.containsKey(layerId)) {
            velocity.put(layerId, weights.copy().scale(0));
        }

        Matrix v = velocity.get(layerId);

        // v = momentum * v - lr * gradient
        Matrix vNew = v.scale(momentum).subtract(gradient.scale(learningRate));

        // weights = weights + v
        Matrix newWeights = weights.add(vNew);

        velocity.put(layerId, vNew);
        return newWeights;
    }

    @Override
    public void reset() {
        velocity.clear();
    }
}