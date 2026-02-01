package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;

public interface Optimizer {
    /**
     * Обновляет веса на основе градиента
     * @param weights текущие веса
     * @param gradient градиент
     * @param layerId идентификатор слоя (для хранения состояния)
     * @return обновлённые веса
     */
    Matrix update(Matrix weights, Matrix gradient, String layerId);

    /**
     * Сбросить состояние оптимизатора
     */
    void reset();
}