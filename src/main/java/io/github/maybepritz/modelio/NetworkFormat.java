package io.github.maybepritz.modelio;

import java.util.List;

public class NetworkFormat {
    public static class Snapshot {
        public Config config;
        public List<LayerMeta> layers;
    }

    public static class Config {
        public double learningRate;
        public double l2Regularization;
        public double dropoutRate;
        public String weightInit;
        public double gradientClip;
        public Long seed;
    }

    public static class LayerMeta {
        public String layerType;        // ← добавлено: "DenseLayer", "SoftmaxLayer", etc.
        public Integer inputSize;       // nullable для слоёв без весов
        public Integer outputSize;
        public String activationClass;
        public Boolean useBias;
        public Double dropoutRate;
    }

    public static class Meta {
        public String version;
        public String framework;
        public long timestamp;
    }
}