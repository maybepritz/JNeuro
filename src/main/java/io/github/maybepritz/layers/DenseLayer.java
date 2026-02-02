package io.github.maybepritz.layers;

import io.github.maybepritz.activations.ActivationFunction;
import io.github.maybepritz.config.LayerConfig;
import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.utils.Matrix;
import io.github.maybepritz.utils.MatrixFactory;

/**
 * Полносвязный (Dense) слой нейронной сети.
 * Выполняет операцию: output = activation(W * input + bias)
 */
public class DenseLayer extends Layer {

    // Обучаемые параметры
    private Matrix weights;
    private Matrix bias;

    // Конфигурация слоя
    private final ActivationFunction activation;
    private final boolean useBias;
    private final double dropoutRate;

    // Кэш для backpropagation
    private Matrix preActivation;  // z = W * input + bias (до активации)
    private Matrix dropoutMask;

    /**
     * Создаёт Dense слой с задан��ой конфигурацией.
     *
     * @param config    конфигурация слоя (размеры, активация)
     * @param netConfig глобальная конфигурация сети
     */
    public DenseLayer(LayerConfig config, NetworkConfig netConfig) {
        int inputSize = config.getInputSize();
        int outputSize = config.getOutputSize();

        this.activation = config.getActivation();
        this.useBias = config.isUseBias();
        this.dropoutRate = config.getDropoutRate() != null
                ? config.getDropoutRate()
                : netConfig.getDropoutRate();

        // Инициализация весов
        this.weights = MatrixFactory.create(outputSize, inputSize);
        initializeWeights(netConfig.getWeightInit(), inputSize, netConfig.getSeed());

        // Инициализация bias (нулями)
        if (useBias) {
            this.bias = MatrixFactory.create(outputSize, 1);
        }
    }

    /**
     * Инициализация весов по выбранной стратегии.
     */
    private void initializeWeights(NetworkConfig.WeightInit init, int fanIn, Long seed) {
        double scale = switch (init) {
            case XAVIER -> Math.sqrt(2.0 / (fanIn + weights.getRows()));
            case HE -> Math.sqrt(2.0 / fanIn);
            case ZEROS -> 0.0;
            default -> 1.0;
        };

        if (scale > 0) {
            weights.randomize(scale, seed);
        }
    }

    // ==================== FORWARD PASS ====================

    /**
     * Прямой проход: вычисляет выход слоя.
     *
     * @param input входной вектор (матрица-столбец)
     * @return выходной вектор после активации
     */
    @Override
    public Matrix forward(Matrix input) {
        this.input = input;

        // 1. Линейное преобразование: z = W * x
        Matrix z = weights.multiply(input);

        // 2. Добавляем bias: z = W * x + b
        if (useBias) {
            z.addColumn(bias);
        }

        // 3. Сохраняем pre-activation для backprop
        this.preActivation = z;

        // 4. Применяем функцию активации: a = f(z)
        this.output = z.map(activation::activate);

        // 5. Dropout (только при обучении)
        if (training && dropoutRate > 0 && dropoutRate < 1.0) {
            applyDropout();
        }

        return output;
    }

    /**
     * Применяет Dropout с инвертированным масштабированием.
     */
    private void applyDropout() {
        double keepProb = 1.0 - dropoutRate;
        dropoutMask = Matrix.randomMask(output.getRows(), output.getCols(), keepProb);
        output = output.elementMultiply(dropoutMask);
        output = output.scale(1.0 / keepProb);  // Inverted dropout
    }

    // ==================== BACKWARD PASS ====================

    @Override
    public boolean isTrainable() {
        return true;
    }

    /**
     * Вычисляет градиент ошибки по выходу слоя.
     * grad = error ⊙ f'(z), где z — pre-activation
     */
    @Override
    public Matrix getGradient(Matrix error, NetworkConfig config) {
        Matrix activationDerivative = preActivation.map(activation::derivative);

        Matrix grad = error.elementMultiply(activationDerivative);

        if (training && dropoutMask != null) {
            grad = grad.elementMultiply(dropoutMask);
        }

        if (config.getGradientClip() > 0) {
            grad.clip(-config.getGradientClip(), config.getGradientClip());
        }

        return grad;
    }

    /**
     * Обратное распространение ошибки и обновление весов.
     *
     * @param error исходная ошибка
     * @param grad  градиент после getGradient()
     * @return ошибка для предыдущего слоя
     */
    @Override
    public Matrix backpropagate(Matrix error, Matrix grad, NetworkConfig config) {
        // 1. Вычисляем градиент по весам: dL/dW = grad * input^T
        Matrix weightGradient = grad.multiply(input.transpose());

        // 2. L2 регуляризация (weight decay)
        if (config.getL2Regularization() > 0) {
            Matrix l2Term = weights.scale(config.getL2Regularization());
            weightGradient = weightGradient.add(l2Term);
        }

        // 3. ВАЖНО: вычисляем ошибку для предыдущего слоя ДО обновления весов!
        Matrix errorForPrevLayer = weights.transpose().multiply(grad);

        // 4. Обновляем веса через оптимизатор
        String layerId = "dense_" + System.identityHashCode(this);
        weights = config.getOptimizer().update(weights, weightGradient, layerId + "_W");

        // 5. Обновляем bias
        if (useBias) {
            bias = config.getOptimizer().update(bias, grad, layerId + "_b");
        }

        return errorForPrevLayer;
    }

    // ==================== GETTERS ====================

    @Override
    public Matrix getWeights() {
        return weights;
    }

    @Override
    public Matrix getBias() {
        return bias;
    }

    @Override
    public ActivationFunction getActivation() {
        return activation;
    }
}