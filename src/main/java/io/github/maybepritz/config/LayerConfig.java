package io.github.maybepritz.config;

import io.github.maybepritz.activations.ActivationFunction;
import io.github.maybepritz.activations.ReLU;

public class LayerConfig {
    /** Количество входных нейронов */
    private final int inputSize;

    /** Количество выходных нейронов */
    private final int outputSize;

    /** Функция активации — добавляет нелинейность */
    private ActivationFunction activation = new ReLU();

    /** Использовать ли bias (смещение) — обычно да */
    private boolean useBias = true;

    /** Dropout для этого слоя (null = берётся из NetworkConfig) */
    private Double dropoutRate = null;

    public LayerConfig(int inputSize, int outputSize){
        if (inputSize <= 0 || outputSize <= 0)
            throw new IllegalArgumentException("Размеры слоя должны быть > 0");

        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public LayerConfig(int inputSize, int outputSize, ActivationFunction activation){
        this(inputSize, outputSize);
        setActivation(activation);
    }

    public int getInputSize() {return this.inputSize;}
    public int getOutputSize() {return this.outputSize;}
    public ActivationFunction getActivation() {return this.activation;}
    public boolean isUseBias() {return this.useBias;}
    public Double getDropoutRate() {return this.dropoutRate;}

    public void setActivation(ActivationFunction activation) {
        if(activation == null)
            throw new IllegalArgumentException("Activation не может быть null");
        this.activation = activation;
    }

    /**
     * @param useBias false только если данные уже центрированы
     */
    public void setUseBias(boolean useBias){
        this.useBias = useBias;
    }

    /**
     * @param rate dropout для этого слоя (null = использовать глобальный)
     */
    public void setDropoutRate(Double rate){
        if(rate != null && (rate < 0 || rate >= 1))
            throw new IllegalArgumentException("Dropout rate должен быть в диапазоне [0, 1)");
        this.dropoutRate = rate;
    }
}
