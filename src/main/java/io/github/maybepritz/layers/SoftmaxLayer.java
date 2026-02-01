package io.github.maybepritz.layers;

import io.github.maybepritz.config.NetworkConfig;
import io.github.maybepritz.utils.Matrix;

public class SoftmaxLayer extends Layer {

    @Override
    public Matrix forward(Matrix input) {
        this.input = input;
        int n = input.getRows();
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = input.get(i, 0);
        }
        double[] y = softmax(x);
        this.output = Matrix.fromArray(y);
        return output;
    }

    @Override
    public Matrix backpropagate(Matrix error, Matrix grad, NetworkConfig config) {
        // Для Softmax + CrossEntropy градиент упрощается
        // Просто пропускаем ошибку дальше
        return error;
    }

    public static double[] softmax(double[] input) {
        // Находим максимум для численной стабильности
        double max = Double.NEGATIVE_INFINITY;
        for (double v : input) {
            if (v > max) max = v;
        }

        // Защита от NaN
        if (Double.isNaN(max) || Double.isInfinite(max)) {
            max = 0;
        }

        double sum = 0.0;
        double[] exp = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            double val = input[i] - max;
            // Ограничиваем чтобы избежать переполнения
            val = Math.max(-500, Math.min(500, val));
            exp[i] = Math.exp(val);
            sum += exp[i];
        }

        // Защита от деления на ноль
        if (sum == 0 || Double.isNaN(sum)) {
            sum = 1e-10;
        }

        for (int i = 0; i < input.length; i++) {
            exp[i] /= sum;
            // Защита от NaN
            if (Double.isNaN(exp[i])) {
                exp[i] = 1.0 / input.length;
            }
        }

        return exp;
    }
}