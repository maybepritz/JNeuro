package io.github.maybepritz.utils;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class MatrixCPU implements Matrix {

    private final double[][] data;
    private final int rows;
    private final int cols;

    public MatrixCPU(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    @Override
    public int getRows() { return rows; }

    @Override
    public int getCols() { return cols; }

    @Override
    public double get(int row, int col) { return data[row][col]; }

    @Override
    public void set(int row, int col, double value) { data[row][col] = value; }

    @Override
    public void randomize(double scale, Long seed) {
        Random rand = (seed != null) ? new Random(seed) : new Random();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = (rand.nextDouble() * 2 - 1) * scale; // [-scale, scale]
    }

    @Override
    public void addColumn(Matrix column) {
        if (column.getRows() != rows || column.getCols() != 1)
            throw new IllegalArgumentException("Column должен быть вектором размера " + rows + "x1");
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] += column.get(i, 0);
    }

    @Override
    public void clip(double min, double max) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = Math.max(min, Math.min(max, data[i][j]));
    }

    @Override
    public Matrix copy() {
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, data[i][j]);
        return result;
    }

    @Override
    public Matrix add(Matrix other) {
        if (rows != other.getRows() || cols != other.getCols())
            throw new IllegalArgumentException("Размеры матриц должны совпадать");
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, data[i][j] + other.get(i, j));
        return result;
    }

    @Override
    public Matrix subtract(Matrix other) {
        if (rows != other.getRows() || cols != other.getCols())
            throw new IllegalArgumentException("Размеры матриц должны совпадать");
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, data[i][j] - other.get(i, j));
        return result;
    }

    @Override
    public Matrix multiply(Matrix other) {
        if (cols != other.getRows())
            throw new IllegalArgumentException("Несовместимые размеры: " + cols + " != " + other.getRows());
        Matrix result = new MatrixCPU(rows, other.getCols());
        for (int i = 0; i < result.getRows(); i++)
            for (int j = 0; j < result.getCols(); j++) {
                double sum = 0;
                for (int k = 0; k < cols; k++)
                    sum += data[i][k] * other.get(k, j);
                result.set(i, j, sum);
            }
        return result;
    }

    @Override
    public Matrix elementMultiply(Matrix other) {
        if (rows != other.getRows() || cols != other.getCols())
            throw new IllegalArgumentException("Размеры матриц должны совпадать");
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, data[i][j] * other.get(i, j));
        return result;
    }

    @Override
    public Matrix scale(double scalar) {
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, data[i][j] * scalar);
        return result;
    }

    @Override
    public Matrix map(DoubleUnaryOperator func) {
        Matrix result = new MatrixCPU(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(i, j, func.applyAsDouble(data[i][j]));
        return result;
    }

    @Override
    public Matrix transpose() {
        Matrix result = new MatrixCPU(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.set(j, i, data[i][j]);
        return result;
    }

    @Override
    public double[] toArray() {
        double[] arr = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                arr[i * cols + j] = data[i][j];
        return arr;
    }
}