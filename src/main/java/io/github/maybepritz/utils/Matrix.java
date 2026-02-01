package io.github.maybepritz.utils;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public interface Matrix {
    int getRows();
    int getCols();

    double get(int row, int col);
    void set(int row, int col, double value);

    Matrix copy();
    Matrix add(Matrix other);
    Matrix subtract(Matrix other);
    Matrix multiply(Matrix other);
    Matrix elementMultiply(Matrix other);
    Matrix scale(double scalar);
    Matrix map(DoubleUnaryOperator func);
    Matrix transpose();
    double[] toArray();

    // Добавленные методы:
    void randomize(double scale, Long seed);
    void addColumn(Matrix column);  // добавляет вектор-столбец к каждому столбцу (broadcast)
    void clip(double min, double max);

    // ==== STATIC factory/utility methods ====

    static Matrix fromArray(double[] arr) {
        Matrix m = MatrixFactory.create(arr.length, 1);
        for (int i = 0; i < arr.length; i++)
            m.set(i, 0, arr[i]);
        return m;
    }

    static Matrix randomMask(int rows, int cols, double keepProb) {
        Matrix mask = MatrixFactory.create(rows, cols);
        Random rand = new Random();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mask.set(i, j, rand.nextDouble() < keepProb ? 1.0 : 0.0);
        return mask;
    }
}