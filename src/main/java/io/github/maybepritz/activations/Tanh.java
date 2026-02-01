package io.github.maybepritz.activations;

public class Tanh implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double tanhX = Math.tanh(x);
        return 1 - tanhX * tanhX;
    }
}