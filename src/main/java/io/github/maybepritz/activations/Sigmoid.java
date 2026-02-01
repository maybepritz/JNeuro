package io.github.maybepritz.activations;

public class Sigmoid implements ActivationFunction{
    @Override
    public double activate(double x){
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x){
        return x * (1.0 - x);
    }
}
