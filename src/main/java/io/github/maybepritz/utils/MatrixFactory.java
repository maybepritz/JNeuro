package io.github.maybepritz.utils;

public class MatrixFactory {
    public enum Backend { CPU, ND4J }
    private static Backend backend = Backend.CPU;

    public static void setBackend(Backend b) { backend = b; }
    public static Backend getBackend() { return backend; }

    public static Matrix create(int rows, int cols) {
        return backend == Backend.ND4J ? new MatrixND4J(rows, cols)
                : new MatrixCPU(rows, cols);
    }
}