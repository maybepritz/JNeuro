package io.github.maybepritz.activations;

public class ReLU implements ActivationFunction {

    @Override
    public double activate(double x) {
        if (Double.isNaN(x)) return 0;
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        if (Double.isNaN(x)) return 0;
        return x > 0 ? 1.0 : 0.0;
    }
}