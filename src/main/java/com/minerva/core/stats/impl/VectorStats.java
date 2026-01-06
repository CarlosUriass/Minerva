package com.minerva.core.stats.impl;

import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;
import java.util.Arrays;

/**
 * Statistical operations specialized for Vector.
 * Can provide optimized implementations for vector-specific operations.
 */
public class VectorStats extends TensorStatsBase {

    private final Vector vector;

    public VectorStats(Vector vector) {
        super(vector);
        this.vector = vector;
    }

    @Override
    public double avg() {
        double[] data = VectorAccess.raw(vector);
        int n = data.length;

        if (n == 0) {
            throw new IllegalAccessError("Vector is empty");
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }

        return sum / n;
    }

    /**
     * Computes the median of the vector.
     * 
     * @return the median value
     */
    public double median() {
        double[] data = VectorAccess.raw(vector);
        int n = data.length;

        boolean odd = n % 2 == 1;
        int mid = n / 2;

        if (odd) {
            return data[mid];
        } else {
            return (data[mid - 1] + data[mid]) / 2.0;
        }
    }

    /**
     * Computes the mode of all elements of the vector.
     * <p>
     * <strong>Note:</strong> Treats floating-point values as discrete.
     * <p>
     * 
     * @return the mode (most frequent value). Returns smallest value if there's a
     *         tie.
     */
    @Override
    public double mode() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        if (n == 0) {
            throw new IllegalArgumentException("Vector is empty");
        }

        if (n == 1)
            return raw[0];

        double[] sorted = Arrays.copyOf(raw, n);
        Arrays.sort(sorted);

        double mode = sorted[0];
        int maxCount = 1;

        double current = sorted[0];
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

        if (count > maxCount) {
            mode = current;
        }

        return mode;
    }

    public double std() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return 0.0;

        double mean = 0.0;
        double m2 = 0.0;
        int count = 0;

        for (int i = 0; i < n; i++) {
            count++;
            double x = raw[i];
            double delta = x - mean;
            mean += delta / count;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }

        // population std
        return Math.sqrt(m2 / n);
    }

    public double max() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        if (n == 0) {
            throw new IllegalArgumentException("Vector is empty");
        }

        double max = raw[0];

        for (int i = 1; i < n; i++) {
            double v = raw[i];
            if (v > max) {
                max = v;
            }
        }

        return max;
    }

    public double min() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        if (n == 0) {
            throw new IllegalArgumentException("Vector is empty");
        }

        double min = raw[0];

        for (int i = 1; i < n; i++) {
            double v = raw[i];
            if (v < min) {
                min = v;
            }
        }
        return min;
    }

    public double sum() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += raw[i];
        }

        return sum;
    }

    public double variance() {
        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;

        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return 0.0;

        double mean = 0.0;
        double m2 = 0.0;
        int count = 0;

        for (int i = 0; i < n; i++) {
            count++;
            double x = raw[i];
            double delta = x - mean;
            mean += delta / count;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }

        // population variance
        return m2 / n;
    }

    /**
     * Computes the specified percentile.
     * <p>
     * Uses linear interpolation (Method 4 in commons-math, R type 7).
     * 
     * @param percentile the percentile (0-100)
     * @return the percentile value
     * @throws IllegalArgumentException if percentile is out of [0, 100] range
     */
    @Override
    public double percentile(double percentile) {
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be between 0 and 100");
        }

        double[] raw = VectorAccess.raw(vector);
        int n = raw.length;
        if (n == 0)
            return Double.NaN;
        if (n == 1)
            return raw[0];

        // 1. Copy and sort (Critical for correctness)
        double[] sorted = Arrays.copyOf(raw, n);
        Arrays.sort(sorted);

        // 2. Compute position
        // Position on 0-based index: (n-1) * p
        double pos = (n - 1) * (percentile / 100.0);

        int lower = (int) pos;
        int upper = lower + 1;
        double weight = pos - lower;

        if (upper >= n) {
            return sorted[n - 1];
        }

        // 3. Linear interpolation
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
    }
}
