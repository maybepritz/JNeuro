package io.github.maybepritz.core;

import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.layers.Layer;
import io.github.maybepritz.utils.Matrix;
import io.github.maybepritz.modelio.NetworkIO;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();
    private final NetworkConfig config;

    public NeuralNetwork(NetworkConfig config){
        this.config = config;
    }

    public NeuralNetwork(){
        this(new NetworkConfig());
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public double[] predict(double[] input){
        setTraining(false);
        return forward(input);
    }

    public double[] forward(double[] input){
        Matrix output = Matrix.fromArray(input);
        for(Layer layer : layers)
            output = layer.forward(output);
        return output.toArray();
    }

    public void train(double[] input, double[] target){
        setTraining(true);
        Matrix output = Matrix.fromArray(input);
        for(Layer layer : layers)
            output = layer.forward(output);
        backpropagate(Matrix.fromArray(target), output);
    }

    public void save(String path) throws IOException {
        NetworkIO.save(this, path);
    }

    public static NeuralNetwork load(String path) throws Exception {
        return NetworkIO.load(path);
    }

    private void backpropagate(Matrix target, Matrix output) {
        Matrix error = target.subtract(output);
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);

            if (!layer.isTrainable()) continue;

            Matrix grad = layer.getGradient(error, config);
            error = layer.backpropagate(error, grad, config);
        }
    }

    private void setTraining(boolean training){
        for(Layer layer : layers)
            layer.setTraining(training);
    }

    public NetworkConfig getConfig() { return config; }
    public List<Layer> getLayers() { return layers; }
}
