package com.minerva.core.stats.impl;

import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import java.util.Arrays;

/**
 * Statistical operations specialized for Matrix.
 * Uses raw array access for maximum performance.
 * Provides both global statistics and axis-specific operations.
 */
public class MatrixStats extends TensorStatsBase {

    private final Matrix matrix;

    public MatrixStats(Matrix matrix) {
        super(matrix);
        this.matrix = matrix;
    }

    // ==================== Global Statistics ====================

    @Override
    public double avg() {
        double[] data = MatrixAccess.raw(matrix);
        int n = data.length;

        if (n == 0)
            throw new IllegalArgumentException("Matrix is empty");

        double sum = 0.0;
        for (double v : data)
            sum += v;
        return sum / n;
    }

    @Override
    public double sum() {
        double[] data = MatrixAccess.raw(matrix);
        double sum = 0.0;
        for (double v : data)
            sum += v;
        return sum;
    }

    @Override
    public double min() {
        double[] data = MatrixAccess.raw(matrix);
        if (data.length == 0)
            throw new IllegalArgumentException("Matrix is empty");

        double min = data[0];
        for (int i = 1; i < data.length; i++)
            if (data[i] < min)
                min = data[i];
        return min;
    }

    @Override
    public double max() {
        double[] data = MatrixAccess.raw(matrix);
        if (data.length == 0)
            throw new IllegalArgumentException("Matrix is empty");

        double max = data[0];
        for (int i = 1; i < data.length; i++)
            if (data[i] > max)
                max = data[i];
        return max;
    }

    @Override
    public double variance() {
        double[] data = MatrixAccess.raw(matrix);
        int n = data.length;

        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return 0.0;

        double mean = 0.0;
        double m2 = 0.0;
        int count = 0;

        for (double x : data) {
            count++;
            double delta = x - mean;
            mean += delta / count;
            m2 += delta * (x - mean);
        }

        return m2 / n; // population variance
    }

    @Override
    public double std() {
        return Math.sqrt(variance());
    }

    @Override
    public double median() {
        double[] data = MatrixAccess.raw(matrix);
        int n = data.length;

        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return data[0];

        double[] sorted = Arrays.copyOf(data, n);
        Arrays.sort(sorted);

        int mid = n / 2;
        return (n % 2 == 1)
                ? sorted[mid]
                : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    @Override
    public double mode() {
        double[] data = MatrixAccess.raw(matrix);
        int n = data.length;

        if (n == 0)
            throw new IllegalArgumentException("Matrix is empty");
        if (n == 1)
            return data[0];

        double[] sorted = Arrays.copyOf(data, n);
        Arrays.sort(sorted);

        double mode = sorted[0];
        double current = sorted[0];
        int maxCount = 1;
        int count = 1;

        for (int i = 1; i < n; i++) {
            if (Double.compare(sorted[i], current) == 0) {
                count++;
            } else {
                if (count > maxCount) {
                    maxCount = count;
                    mode = current;
                }
                current = sorted[i];
                count = 1;
            }
        }

        if (count > maxCount)
            mode = current;

        return mode;
    }

    @Override
    public double percentile(double percentile) {
        if (percentile < 0 || percentile > 100)
            throw new IllegalArgumentException("Percentile must be between 0 and 100");

        double[] data = MatrixAccess.raw(matrix);
        int n = data.length;

        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return data[0];

        double[] sorted = Arrays.copyOf(data, n);
        Arrays.sort(sorted);

        double pos = (n - 1) * (percentile / 100.0);
        int lower = (int) pos;
        int upper = lower + 1;
        double weight = pos - lower;

        if (upper >= n)
            return sorted[n - 1];

        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
    }

    // ==================== Axis-wise Operations ====================

    public Vector mean(int axis) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] data = MatrixAccess.raw(matrix);

        if (axis == 0) { // column-wise
            double[] out = new double[cols];
            for (int c = 0; c < cols; c++) {
                double sum = 0.0;
                for (int r = 0; r < rows; r++)
                    sum += data[r * cols + c];
                out[c] = sum / rows;
            }
            return new Vector(out);
        }

        if (axis == 1) { // row-wise
            double[] out = new double[rows];
            for (int r = 0; r < rows; r++) {
                double sum = 0.0;
                int base = r * cols;
                for (int c = 0; c < cols; c++)
                    sum += data[base + c];
                out[r] = sum / cols;
            }
            return new Vector(out);
        }

        throw new IllegalArgumentException("Axis must be 0 or 1");
    }

    public Vector sum(int axis) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] data = MatrixAccess.raw(matrix);

        if (axis == 0) {
            double[] out = new double[cols];
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++)
                    out[c] += data[r * cols + c];
            return new Vector(out);
        }

        if (axis == 1) {
            double[] out = new double[rows];
            for (int r = 0; r < rows; r++) {
                int base = r * cols;
                for (int c = 0; c < cols; c++)
                    out[r] += data[base + c];
            }
            return new Vector(out);
        }

        throw new IllegalArgumentException("Axis must be 0 or 1");
    }

    public Vector min(int axis) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] data = MatrixAccess.raw(matrix);

        if (axis == 0) {
            double[] out = new double[cols];
            Arrays.fill(out, Double.POSITIVE_INFINITY);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    out[c] = Math.min(out[c], data[r * cols + c]);
            return new Vector(out);
        }

        if (axis == 1) {
            double[] out = new double[rows];
            for (int r = 0; r < rows; r++) {
                double min = data[r * cols];
                int base = r * cols;
                for (int c = 1; c < cols; c++)
                    min = Math.min(min, data[base + c]);
                out[r] = min;
            }
            return new Vector(out);
        }

        throw new IllegalArgumentException("Axis must be 0 or 1");
    }

    public Vector max(int axis) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] data = MatrixAccess.raw(matrix);

        if (axis == 0) {
            double[] out = new double[cols];
            Arrays.fill(out, Double.NEGATIVE_INFINITY);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    out[c] = Math.max(out[c], data[r * cols + c]);
            return new Vector(out);
        }

        if (axis == 1) {
            double[] out = new double[rows];
            for (int r = 0; r < rows; r++) {
                double max = data[r * cols];
                int base = r * cols;
                for (int c = 1; c < cols; c++)
                    max = Math.max(max, data[base + c]);
                out[r] = max;
            }
            return new Vector(out);
        }

        throw new IllegalArgumentException("Axis must be 0 or 1");
    }

    public Vector std(int axis) {
        Vector mean = mean(axis);
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] data = MatrixAccess.raw(matrix);

        if (axis == 0) {
            double[] out = new double[cols];
            for (int c = 0; c < cols; c++) {
                double m = mean.get(c);
                double acc = 0.0;
                for (int r = 0; r < rows; r++) {
                    double d = data[r * cols + c] - m;
                    acc += d * d;
                }
                out[c] = Math.sqrt(acc / rows);
            }
            return new Vector(out);
        }

        if (axis == 1) {
            double[] out = new double[rows];
            for (int r = 0; r < rows; r++) {
                double m = mean.get(r);
                double acc = 0.0;
                int base = r * cols;
                for (int c = 0; c < cols; c++) {
                    double d = data[base + c] - m;
                    acc += d * d;
                }
                out[r] = Math.sqrt(acc / cols);
            }
            return new Vector(out);
        }

        throw new IllegalArgumentException("Axis must be 0 or 1");
    }
}
