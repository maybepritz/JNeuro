package io.github.maybepritz.layers;

import io.github.maybepritz.activations.ActivationFunction;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.utils.Matrix;

public abstract class Layer {
    protected Matrix input;
    protected Matrix output;
    protected boolean training = true;

    public abstract Matrix forward(Matrix input);

    // Для обратного прохода (по умолчанию не обучаемый слой)
    public boolean isTrainable() { return false; }

    // Вызывается только если isTrainable() == true
    public Matrix getGradient(Matrix error, NetworkConfig config) { return null; }
    public Matrix backpropagate(Matrix error, Matrix grad, NetworkConfig config) { return null; }

    public void setTraining(boolean training) { this.training = training; }
    public Matrix getOutput() { return output; }
    public Matrix getInput() { return input; }

    // Методы для сериализации (переопределяются в обучаемых слоях)
    public Matrix getWeights() { return null; }
    public Matrix getBias() { return null; }
    public ActivationFunction getActivation() { return null; }
}