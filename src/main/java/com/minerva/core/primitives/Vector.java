package com.minerva.core.primitives;

import com.minerva.core.api.ITensor;
import com.minerva.core.api.IVectorOps;
import com.minerva.core.stats.impl.VectorStats;

import java.util.Arrays;

/**
 * A one-dimensional array of doubles.
 * Represents a mathematical vector with support for common linear algebra
 * operations.
 */
public class Vector implements ITensor, IVectorOps {

    // Package-private for VectorAccess (friend-like pattern)
    final double[] data;

    // Lazily initialized statistics wrapper
    private transient VectorStats stats;

    /**
     * Creates a vector of the specified size initialized to zeros.
     *
     * @param size the number of elements
     */
    public Vector(int size) {
        this.data = new double[size];
    }

    /**
     * Creates a vector of the specified size filled with the given value.
     *
     *
     * @param size  the number of elements
     * @param value the value to fill with
     */
    public Vector(int size, double value) {
        this.data = new double[size];
        if (value != 0.0)
            java.util.Arrays.fill(this.data, value);
    }

    /**
     * Creates a vector from an existing array.
     * Performs a defensive copy.
     *
     * @param data the initial values
     */
    public Vector(double[] data) {
        this.data = data.clone();
    }

    // ==================== Tensor Interface ====================

    @Override
    public int[] shape() {
        return new int[] { data.length };
    }

    @Override
    public int size() {
        return data.length;
    }

    @Override
    public double get(int... indices) {
        if (indices.length != 1) {
            throw new IllegalArgumentException("Vector requires exactly 1 index");
        }
        int i = indices[0];
        if (i < 0 || i >= data.length) {
            throw new IndexOutOfBoundsException("Index " + i + " out of bounds");
        }
        return data[i];
    }

    @Override
    public void set(double value, int... indices) {
        if (indices.length != 1) {
            throw new IllegalArgumentException("Vector requires exactly 1 index");
        }
        int i = indices[0];
        if (i < 0 || i >= data.length) {
            throw new IndexOutOfBoundsException("Index " + i + " out of bounds");
        }
        data[i] = value;
    }

    // ==================== VectorOps Interface ====================

    @Override
    public double dot(Vector other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException("Vector must have the same size for dot product");
        }

        double sum = 0.0;
        int n = data.length;

        for (int i = 0; i < n; i++) {
            sum += this.data[i] * other.data[i];
        }

        return sum;
    }

    @Override
    public Vector add(Vector other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException("Vector must have the same size for addition");
        }

        int n = data.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            result[i] = this.data[i] + other.data[i];
        }

        return new Vector(result);
    }

    @Override
    public Vector subtract(Vector other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException("Vector must have the same size for subtraction");
        }

        int n = data.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            result[i] = this.data[i] - other.data[i];
        }

        return new Vector(result);
    }

    @Override
    public Vector scale(double scalar) {
        int n = data.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            result[i] = this.data[i] * scalar;
        }

        return new Vector(result);
    }

    @Override
    public double norm() {
        double squares = 0.0;

        for (double v : data) {
            squares += v * v;
        }

        return Math.sqrt(squares);
    }

    // ==================== Statistics API ====================

    /**
     * Returns a statistics wrapper for this vector.
     * The returned instance is cached and recomputed on demand.
     *
     * @return a VectorStats instance for statistical computations
     */
    public VectorStats stats() {
        if (stats == null) {
            stats = new VectorStats(this);
        }
        return stats;
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }
}
