package io.github.maybepritz.activations;

public interface ActivationFunction {
    double activate(double x);
    double derivative(double x);
}
