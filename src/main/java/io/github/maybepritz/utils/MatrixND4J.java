package io.github.maybepritz.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.DoubleUnaryOperator;

public class MatrixND4J implements Matrix {
    private INDArray data;

    public MatrixND4J(int rows, int cols) {
        this.data = Nd4j.zeros(rows, cols);
    }

    public MatrixND4J(INDArray data) {
        this.data = data;
    }

    @Override
    public int getRows() { return (int) data.rows(); }

    @Override
    public int getCols() { return (int) data.columns(); }

    @Override
    public double get(int row, int col) { return data.getDouble(row, col); }

    @Override
    public void set(int row, int col, double value) { data.putScalar(row, col, value); }

    @Override
    public Matrix multiply(Matrix other) {
        INDArray otherData = ((MatrixND4J) other).data;
        return new MatrixND4J(data.mmul(otherData));
    }

    @Override
    public Matrix add(Matrix other) {
        INDArray otherData = ((MatrixND4J) other).data;
        return new MatrixND4J(data.add(otherData));
    }

    @Override
    public Matrix subtract(Matrix other) {
        INDArray otherData = ((MatrixND4J) other).data;
        return new MatrixND4J(data.sub(otherData));
    }

    @Override
    public Matrix elementMultiply(Matrix other) {
        INDArray otherData = ((MatrixND4J) other).data;
        return new MatrixND4J(data.mul(otherData));
    }

    @Override
    public Matrix transpose() {
        return new MatrixND4J(data.transpose());
    }

    @Override
    public Matrix scale(double scalar) {
        return new MatrixND4J(data.mul(scalar));
    }

    @Override
    public Matrix map(DoubleUnaryOperator fn) {
        // Для стандартных функций используй встроенные
        // Это медленная операция - избегай если возможно
        INDArray result = data.dup();
        for (int i = 0; i < result.length(); i++) {
            result.putScalar(i, fn.applyAsDouble(result.getDouble(i)));
        }
        return new MatrixND4J(result);
    }

    // Оптимизированные версии для стандартных активаций
    public Matrix sigmoid() {
        return new MatrixND4J(Transforms.sigmoid(data, true));
    }

    public Matrix relu() {
        return new MatrixND4J(Transforms.relu(data, true));
    }

    public Matrix tanh() {
        return new MatrixND4J(Transforms.tanh(data, true));
    }

    @Override
    public void addColumn(Matrix column) {
        INDArray colData = ((MatrixND4J) column).data;
        data.addiColumnVector(colData);
    }

    @Override
    public Matrix copy() {
        return new MatrixND4J(data.dup());
    }

    @Override
    public void randomize(double scale, Long seed) {
        if (seed != null) Nd4j.getRandom().setSeed(seed);
        this.data = Nd4j.randn(data.rows(), data.columns()).mul(scale);
    }

    @Override
    public void clip(double min, double max) {
        Transforms.min(data, max, false);
        Transforms.max(data, min, false);
    }

    @Override
    public double[] toArray() {
        return data.data().asDouble();
    }

    // Геттер для прямого доступа
    public INDArray getData() { return data; }
}