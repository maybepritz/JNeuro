package io.github.maybepritz.config;

public class TrainingConfig {
    /** Количество эпох — сколько раз пройти по всему датасету */
    private int epochs = 1000;

    /** Размер батча — сколько примеров обрабатывать до обновления весов */
    private int batchSize = 32;

    /** Перемешивать ли данные каждую эпоху */
    private boolean shuffle = true;

    /** Включить ли раннюю остановку при отсутствии улучшений */
    private boolean earlyStopping = false;

    /** Сколько эпох ждать без улучшений перед остановкой */
    private int patience = 10;

    /** Минимальное улучшение loss, чтобы считать прогресс */
    private double minDelta = 0.0001;

    /** Как часто выводить лог (каждые N эпох) */
    private int logInterval = 100;

    /** Выводить ли логи в консоль */
    private boolean verbose = true;

    /** Доля данных для валидации (0.0 = без валидации) */
    private double validationSplit = 0.0;

    private int numThreads = 0;

    public int getEpochs() {return this.epochs;}
    public int getBatchSize() {return this.batchSize;}
    public boolean isShuffle() {return this.shuffle;}
    public boolean isEarlyStopping() {return this.earlyStopping;}
    public int getPatience() {return this.patience;}
    public double getMinDelta() {return this.minDelta;}
    public int getLogInterval() {return this.logInterval;}
    public boolean isVerbose() {return this.verbose;}
    public double getValidationSplit() {return this.validationSplit;}
    public int getNumThreads() { return this.numThreads;}

    /**
     * @param epochs количество проходов по датасету
     */
    public void setEpochs(int epochs) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs должен быть > 0");
        }
        this.epochs = epochs;
    }

    /**
     * @param batchSize размер мини-батча (1 = SGD, весь датасет = batch GD)
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size должен быть > 0");
        }
        this.batchSize = batchSize;
    }

    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    public void setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
    }

    /**
     * @param patience эпохи ожидания (рекомендуется 5-20)
     */
    public void setPatience(int patience) {
        if (patience <= 0) {
            throw new IllegalArgumentException("Patience должен быть > 0");
        }
        this.patience = patience;
    }

    /**
     * @param minDelta порог улучшения (рекомендуется 0.0001 - 0.001)
     */
    public void setMinDelta(double minDelta) {
        if (minDelta < 0) {
            throw new IllegalArgumentException("MinDelta не может быть отрицательным");
        }
        this.minDelta = minDelta;
    }

    public void setLogInterval(int logInterval) {
        if (logInterval <= 0) {
            throw new IllegalArgumentException("Log interval должен быть > 0");
        }
        this.logInterval = logInterval;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * @param split доля валидации (рекомендуется 0.1 - 0.2)
     */
    public void setValidationSplit(double split) {
        if (split < 0 || split >= 1) {
            throw new IllegalArgumentException("Validation split должен быть в диапазоне [0, 1)");
        }
        this.validationSplit = split;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

}
