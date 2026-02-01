package io.github.maybepritz.config;

public class NetworkConfig {
    public enum WeightInit{
        /** Случайные значения в диапазоне [-1, 1] */
        RANDOM,
        /** Xavier/Glorot — хорош для Sigmoid, Tanh */
        XAVIER,
        /** He — оптимален для ReLU */
        HE,
        /** Все веса = 0 (обычно не рекомендуется) */
        ZEROS
    }

    /** Скорость обучения — насколько сильно корректируются веса за шаг */
    private double learningRate = 0.01;

    /** L2 регуляризация — штраф за большие веса, предотвращает переобучение */
    private double l2Regularization = 0.0;

    /** Dropout — доля нейронов, отключаемых при обучении (0.0 = выключено) */
    private double dropoutRate = 0.0;

    /** Способ инициализации весов */
    private WeightInit weightInit = WeightInit.RANDOM;

    /** Обрезка градиентов — предотвращает "взрыв" градиентов (0.0 = выключено) */
    private double gradientClip = 0.0;

    /** Seed для генератора случайных чисел — для воспроизводимости результатов */
    private Long seed = null;

    public double getLearningRate() {return this.learningRate;}
    public double getL2Regularization() {return this.l2Regularization;}
    public double getDropoutRate() {return this.dropoutRate;}
    public WeightInit getWeightInit() {return this.weightInit;}
    public double getGradientClip() {return this.gradientClip;}
    public Long getSeed() {return this.seed;}

    /**
     * @param learningRate скорость обучения (рекомендуется 0.001 - 0.1)
     */
    public void setLearningRate(double learningRate){
        if(learningRate <= 0)
            throw new IllegalArgumentException("Learning rate должен быть > 0");
        this.learningRate = learningRate;
    }

    /**
     * @param l2 коэффициент L2 регуляризации (рекомендуется 0.0001 - 0.01)
     */
    public void setL2Regularization(double l2){
        if(l2 < 0)
            throw new IllegalArgumentException("L2 регуляризация не может быть отрицательной");
        this.l2Regularization = l2;
    }

    /**
     * @param rate доля отключаемых нейронов (рекомендуется 0.1 - 0.5)
     */
    public void setDropoutRate(double rate) {
        if(rate < 0 || rate >= 1)
            throw new IllegalArgumentException("Dropout rate должен быть в диапазоне [0, 1)");
        this.dropoutRate = rate;
    }

    public void setWeightInit(WeightInit weightInit){
        if(weightInit == null)
            throw new IllegalArgumentException("WeightInit не может быть null");
        this.weightInit = weightInit;
    }

    /**
     * @param clip максимальное абсолютное значение градиента (рекомендуется 1.0 - 5.0)
     */
    public void setGradientClip(double clip) {
        if(clip < 0)
            throw new IllegalArgumentException("Gradient clip не может быть отрицательным");
        this.gradientClip = clip;
    }

    /**
     * @param seed seed для воспроизводимости (null = случайный)
     */
    public void setSeed(Long seed){
        this.seed = seed;
    }
}
