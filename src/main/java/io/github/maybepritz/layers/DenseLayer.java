package io.github.maybepritz.layers;

import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.utils.Matrix;
import io.github.maybepritz.utils.MatrixFactory;
import io.github.maybepritz.activations.ActivationFunction;

public class DenseLayer extends Layer {
    private Matrix weights;
    private Matrix bias;
    private final ActivationFunction activation;
    private final boolean useBias;
    private final double dropoutRate;
    private Matrix dropoutMask;

    public DenseLayer(LayerConfig config, NetworkConfig netConfig) {
        int inputSize = config.getInputSize();
        int outputSize = config.getOutputSize();
        this.activation = config.getActivation();
        this.useBias = config.isUseBias();
        this.dropoutRate = (config.getDropoutRate() != null) ? config.getDropoutRate() : netConfig.getDropoutRate();

        this.weights = MatrixFactory.create(outputSize, inputSize);
        initializeWeights(netConfig.getWeightInit(), inputSize, outputSize, netConfig.getSeed());

        if (useBias) {
            this.bias = MatrixFactory.create(outputSize, 1);
        }
    }

    private void initializeWeights(NetworkConfig.WeightInit init, int fanIn, int fanOut, Long seed) {
        switch (init) {
            case XAVIER -> weights.randomize(Math.sqrt(2.0 / (fanIn + fanOut)), seed);
            case HE -> weights.randomize(Math.sqrt(2.0 / fanIn), seed);
            case ZEROS -> { /* leave zeros */ }
            default -> weights.randomize(1.0, seed);
        }
    }

    @Override
    public Matrix forward(Matrix input) {
        this.input = input;
        Matrix z = weights.multiply(input);

        if (useBias) {
            z.addColumn(bias);
        }

        this.output = z.map(activation::activate);

        if (training && dropoutRate > 0) {
            dropoutMask = Matrix.randomMask(output.getRows(), output.getCols(), 1 - dropoutRate);
            output = output.elementMultiply(dropoutMask);
            output = output.scale(1.0 / (1 - dropoutRate));
        }
        return output;
    }

    @Override
    public boolean isTrainable() { return true; }

    @Override
    public Matrix getGradient(Matrix error, NetworkConfig config) {
        Matrix grad = output.map(activation::derivative);
        grad = grad.elementMultiply(error);
        grad = grad.scale(config.getLearningRate());
        if (config.getGradientClip() > 0) {
            grad.clip(-config.getGradientClip(), config.getGradientClip());
        }
        return grad;
    }

    @Override
    public Matrix backpropagate(Matrix error, Matrix grad, NetworkConfig config) {
        Matrix inputT = input.transpose();
        Matrix weightsDelta = grad.multiply(inputT);

        // L2 регуляризация
        if (config.getL2Regularization() > 0) {
            Matrix l2Penalty = weights.copy();
            l2Penalty = l2Penalty.scale(config.getL2Regularization() * config.getLearningRate());
            weightsDelta = weightsDelta.subtract(l2Penalty);
        }

        double clipValue = 5.0;
        for (int i = 0; i < weightsDelta.getRows(); i++) {
            for (int j = 0; j < weightsDelta.getCols(); j++) {
                double val = weightsDelta.get(i, j);
                if (Double.isNaN(val)) val = 0;
                val = Math.max(-clipValue, Math.min(clipValue, val));
                weightsDelta.set(i, j, val);
            }
        }

        weights = weights.add(weightsDelta);
        if (useBias) bias = bias.add(grad);

        Matrix transposedWeights = weights.transpose();
        return transposedWeights.multiply(error);
    }

    // Геттеры:
    public Matrix getWeights() { return weights; }
    public Matrix getBias() { return bias; }
    public ActivationFunction getActivation() { return activation; }
}