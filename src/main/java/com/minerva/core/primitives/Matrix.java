package com.minerva.core.primitives;

import com.minerva.core.api.ITensor;
import com.minerva.core.api.IMatrixOps;
import com.minerva.core.stats.impl.MatrixStats;

/**
 * A two-dimensional array of doubles.
 * Represents a mathematical matrix with support for common linear algebra
 * operations.
 *
 * Internal storage is row-major and contiguous for performance.
 */
public class Matrix implements ITensor, IMatrixOps {

    // Package-private for friend-like access (MatrixAccess, stats, ops)
    final int rows;
    final int cols;
    final double[] data;

    private transient MatrixStats stats;

    /**
     * Creates a matrix of the specified dimensions initialized to zeros.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0)
            throw new IllegalArgumentException("Rows and columns must be positive");
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    /**
     * Creates a matrix of the specified dimensions filled with the given value.
     * Uses Arrays.fill() for maximum performance.
     *
     * @param rows  number of rows
     * @param cols  number of columns
     * @param value the value to fill with
     */
    public Matrix(int rows, int cols, double value) {
        if (rows <= 0 || cols <= 0)
            throw new IllegalArgumentException("Rows and columns must be positive");
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
        if (value != 0.0)
            java.util.Arrays.fill(this.data, value);
    }

    /**
     * Creates a matrix from an existing 2D array.
     * Performs a defensive copy into a contiguous internal layout.
     *
     * @param data the initial values
     */
    public Matrix(double[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("Data cannot be null or empty");

        this.rows = data.length;
        this.cols = data[0].length;

        if (cols == 0)
            throw new IllegalArgumentException("Matrix must have at least one column");

        this.data = new double[rows * cols];

        for (int r = 0; r < rows; r++) {
            if (data[r].length != cols)
                throw new IllegalArgumentException("All rows must have the same length");
            System.arraycopy(data[r], 0, this.data, r * cols, cols);
        }
    }

    // Internal constructor for zero-copy ops
    Matrix(int rows, int cols, double[] data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    // ==================== Tensor Interface ====================

    @Override
    public int[] shape() {
        return new int[] { rows, cols };
    }

    @Override
    public int size() {
        return rows * cols;
    }

    @Override
    public double get(int... indices) {
        if (indices.length != 2) {
            throw new IllegalArgumentException("Matrix requires exactly 2 indices (row, col)");
        }
        return get(indices[0], indices[1]);
    }

    @Override
    public void set(double value, int... indices) {
        if (indices.length != 2) {
            throw new IllegalArgumentException("Matrix requires exactly 2 indices (row, col)");
        }
        set(value, indices[0], indices[1]);
    }

    // Strongly-typed accessors (preferred internally)

    public double get(int row, int col) {
        checkBounds(row, col);
        return data[row * cols + col];
    }

    public void set(double value, int row, int col) {
        checkBounds(row, col);
        data[row * cols + col] = value;
    }

    private void checkBounds(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw new IndexOutOfBoundsException(
                    "Index [" + r + ", " + c + "] out of bounds for shape [" + rows + ", " + cols + "]");
        }
    }

    // ==================== MatrixOps Interface ====================

    @Override
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimension mismatch: [" + rows + "x" + cols + "] x [" +
                            other.rows + "x" + other.cols + "]");
        }

        Matrix result = new Matrix(this.rows, other.cols);
        double[] res = result.data;
        double[] a = this.data;
        double[] b = other.data;

        int m = this.rows;
        int n = this.cols;
        int p = other.cols;

        // Cache-friendly loop order: i-k-j
        for (int i = 0; i < m; i++) {
            int aRow = i * n;
            int rRow = i * p;
            for (int k = 0; k < n; k++) {
                double aik = a[aRow + k];
                int bRow = k * p;
                for (int j = 0; j < p; j++) {
                    res[rRow + j] += aik * b[bRow + j];
                }
            }
        }
        return result;
    }

    @Override
    public Vector multiply(Vector vector) {
        if (this.cols != vector.size()) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: Matrix columns " + cols + " != Vector size " + vector.size());
        }

        Vector result = new Vector(this.rows);
        double[] res = VectorAccess.raw(result);
        double[] v = VectorAccess.raw(vector);

        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            int rowOffset = i * cols;
            for (int j = 0; j < cols; j++) {
                sum += data[rowOffset + j] * v[j];
            }
            res[i] = sum;
        }
        return result;
    }

    @Override
    public Matrix add(Matrix other) {
        checkDimensions(other);
        double[] res = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] + other.data[i];
        }
        return new Matrix(rows, cols, res);
    }

    @Override
    public Matrix subtract(Matrix other) {
        checkDimensions(other);
        double[] res = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = this.data[i] - other.data[i];
        }
        return new Matrix(rows, cols, res);
    }

    @Override
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        double[] res = result.data;

        for (int r = 0; r < rows; r++) {
            int rowOffset = r * cols;
            for (int c = 0; c < cols; c++) {
                res[c * rows + r] = data[rowOffset + c];
            }
        }
        return result;
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    private void checkDimensions(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }
    }

    /**
     * Creates an identity matrix of the specified size.
     * The identity matrix has 1s on the diagonal and 0s elsewhere.
     *
     * @param rows the number of rows (and columns) for the square matrix
     * @return a new identity matrix
     */
    public static Matrix identity(int rows) {
        double[] data = new double[rows * rows];

        for (int i = 0; i < rows; i++) {
            data[i * (rows + 1)] = 1.0;
        }

        return new Matrix(rows, rows, data);
    }

    // ==================== Statistics API ====================

    public MatrixStats stats() {
        if (stats == null)
            stats = new MatrixStats(this);
        return stats;
    }

    // ==================== Object Methods ====================

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int r = 0; r < rows; r++) {
            sb.append("[ ");
            for (int c = 0; c < cols; c++) {
                sb.append(String.format("%8.4f", data[r * cols + c]));
                if (c < cols - 1)
                    sb.append(", ");
            }
            sb.append(" ]\n");
        }
        return sb.toString();
    }
}
