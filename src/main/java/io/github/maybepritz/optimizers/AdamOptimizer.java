package io.github.maybepritz.optimizers;

import io.github.maybepritz.utils.Matrix;
import io.github.maybepritz.utils.MatrixFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Adam (Adaptive Moment Estimation) оптимизатор.
 *
 * Комбинирует идеи Momentum и RMSprop:
 * - m (первый момент): скользящее среднее градиентов
 * - v (второй момент): скользящее среднее квадратов градиентов
 *
 * Формулы:
 *   m_t = β1 * m_{t-1} + (1 - β1) * g_t
 *   v_t = β2 * v_{t-1} + (1 - β2) * g_t²
 *   m̂_t = m_t / (1 - β1^t)  // bias correction
 *   v̂_t = v_t / (1 - β2^t)  // bias correction
 *   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
 */
public class AdamOptimizer implements Optimizer {

    // Гиперпараметры
    private final double learningRate;  // α (обычно 0.001)
    private final double beta1;         // β1 (обычно 0.9)
    private final double beta2;         // β2 (обычно 0.999)
    private final double epsilon;       // ε (обычно 1e-8)

    // Состояние для каждого параметра (по layerId)
    private final Map<String, Matrix> m;  // Первый момент (momentum)
    private final Map<String, Matrix> v;  // Второй момент (velocity)
    private final Map<String, Integer> t; // Счётчик шагов

    /**
     * Создаёт Adam с learning rate и параметрами по умолчанию.
     */
    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    /**
     * Создаёт Adam с полной настройкой параметров.
     */
    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        // ConcurrentHashMap для thread-safety при многопоточном обучении
        this.m = new ConcurrentHashMap<>();
        this.v = new ConcurrentHashMap<>();
        this.t = new ConcurrentHashMap<>();
    }

    /**
     * Обновляет веса по алгоритму Adam.
     *
     * @param weights  текущие веса
     * @param gradient градиент по весам
     * @param layerId  уникальный идентификатор слоя
     * @return обновлённые веса
     */
    @Override
    public Matrix update(Matrix weights, Matrix gradient, String layerId) {
        // Инициализация состояния при первом вызове
        initializeIfNeeded(layerId, weights.getRows(), weights.getCols());

        // Увеличиваем счётчик шагов
        int timestep = t.get(layerId) + 1;
        t.put(layerId, timestep);

        // Получаем текущее состояние
        Matrix mCurrent = m.get(layerId);
        Matrix vCurrent = v.get(layerId);

        // Обновляем первый момент: m = β1 * m + (1 - β1) * g
        Matrix mNew = updateFirstMoment(mCurrent, gradient);

        // Обновляем второй момент: v = β2 * v + (1 - β2) * g²
        Matrix vNew = updateSecondMoment(vCurrent, gradient);

        // Bias correction
        Matrix mHat = biasCorrection(mNew, beta1, timestep);
        Matrix vHat = biasCorrection(vNew, beta2, timestep);

        // Вычисляем обновление: Δw = α * m̂ / (√v̂ + ε)
        Matrix update = computeUpdate(mHat, vHat);

        // Обновляем веса: w = w - Δw
        Matrix newWeights = weights.subtract(update);

        // Сохраняем новое состояние
        m.put(layerId, mNew);
        v.put(layerId, vNew);

        return newWeights;
    }

    /**
     * Инициализирует состояние для нового слоя.
     */
    private void initializeIfNeeded(String layerId, int rows, int cols) {
        if (!m.containsKey(layerId)) {
            // Создаём нулевые матрицы нужного размера
            m.put(layerId, MatrixFactory.create(rows, cols));
            v.put(layerId, MatrixFactory.create(rows, cols));
            t.put(layerId, 0);
        }
    }

    /**
     * m_t = β1 * m_{t-1} + (1 - β1) * g_t
     */
    private Matrix updateFirstMoment(Matrix mCurrent, Matrix gradient) {
        Matrix scaledM = mCurrent.scale(beta1);
        Matrix scaledG = gradient.scale(1.0 - beta1);
        return scaledM.add(scaledG);
    }

    /**
     * v_t = β2 * v_{t-1} + (1 - β2) * g_t²
     */
    private Matrix updateSecondMoment(Matrix vCurrent, Matrix gradient) {
        Matrix scaledV = vCurrent.scale(beta2);
        Matrix gSquared = gradient.elementMultiply(gradient);
        Matrix scaledGSquared = gSquared.scale(1.0 - beta2);
        return scaledV.add(scaledGSquared);
    }

    /**
     * Bias correction: x̂ = x / (1 - β^t)
     */
    private Matrix biasCorrection(Matrix moment, double beta, int timestep) {
        double correction = 1.0 / (1.0 - Math.pow(beta, timestep));
        return moment.scale(correction);
    }

    /**
     * Δw = α * m̂ / (√v̂ + ε)
     */
    private Matrix computeUpdate(Matrix mHat, Matrix vHat) {
        // Вычисляем 1 / (√v̂ + ε) для каждого элемента
        Matrix invSqrtV = vHat.map(x -> 1.0 / (Math.sqrt(x) + epsilon));

        // Δw = α * m̂ ⊙ invSqrtV
        return mHat.elementMultiply(invSqrtV).scale(learningRate);
    }

    /**
     * Сбрасывает состояние оптимизатора.
     * Вызывается при начале нового обучения.
     */
    @Override
    public void reset() {
        m.clear();
        v.clear();
        t.clear();
    }
}