package io.github.maybepritz.activations;

public class LeakyReLU implements ActivationFunction{
    private final double scale;

    public LeakyReLU(){
        this(0.01);
    }

    public LeakyReLU(double scale){
        this.scale = scale;
    }

    @Override
    public double activate(double x){
        return x >= 0 ? x : scale * x;
    }

    @Override
    public double derivative(double x){
        return x >= 0 ? 1.0 : scale;
    }
}
