package com.minerva.models.regression.impl;

import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;
import com.minerva.models.regression.api.IRegressionModel;

/**
 * Ordinary Least Squares Linear Regression using QR Decomposition
 * (Modified Gram-Schmidt).
 *
 * Solves:
 *   min ||X w - y||²
 *
 * with implicit intercept.
 *
 * Numerically stable, cache-friendly, and dimension-safe.
 */
public final class LinearRegression implements IRegressionModel {

    /** Learned weights: [intercept, w1, w2, ...] */
    private Vector weights;
    private boolean fitted = false;

    @Override
    public void fit(Matrix X, Vector y) {
        final int n = X.rows();
        final int p = X.cols();
        final int k = p + 1; // intercept + features

        if (n != y.size()) {
            throw new IllegalArgumentException("X rows must match y length");
        }

        // Raw data
        final double[] xRaw = MatrixAccess.raw(X);
        final double[] yRaw = VectorAccess.raw(y);

        // Build augmented design matrix X̃ = [1 | X]
        double[][] Q = new double[n][k];
        for (int i = 0; i < n; i++) {
            Q[i][0] = 1.0; // intercept
            System.arraycopy(xRaw, i * p, Q[i], 1, p);
        }

        // R upper triangular
        double[][] R = new double[k][k];

        // Modified Gram–Schmidt QR
        for (int j = 0; j < k; j++) {

            // Compute norm
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += Q[i][j] * Q[i][j];
            }
            norm = Math.sqrt(norm);

            if (norm < 1e-12) {
                throw new ArithmeticException("Rank deficient matrix");
            }

            R[j][j] = norm;

            // Normalize column
            for (int i = 0; i < n; i++) {
                Q[i][j] /= norm;
            }

            // Orthogonalize remaining columns
            for (int l = j + 1; l < k; l++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++) {
                    dot += Q[i][j] * Q[i][l];
                }
                R[j][l] = dot;
                for (int i = 0; i < n; i++) {
                    Q[i][l] -= dot * Q[i][j];
                }
            }
        }

        // Compute Qᵀy
        double[] qty = new double[k];
        for (int j = 0; j < k; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += Q[i][j] * yRaw[i];
            }
            qty[j] = sum;
        }

        // Back substitution: R w = Qᵀ y
        double[] w = new double[k];
        for (int i = k - 1; i >= 0; i--) {
            double sum = qty[i];
            for (int j = i + 1; j < k; j++) {
                sum -= R[i][j] * w[j];
            }
            w[i] = sum / R[i][i];
        }

        this.weights = new Vector(w);
        this.fitted = true;
    }

    @Override
    public Vector predict(Matrix X) {
        if (!fitted) {
            throw new IllegalStateException("Model not fitted");
        }

        final int n = X.rows();
        final int p = X.cols();

        double[] x = MatrixAccess.raw(X);
        double[] w = VectorAccess.raw(weights);
        double[] yPred = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = w[0]; // intercept
            int row = i * p;
            for (int j = 0; j < p; j++) {
                sum += x[row + j] * w[j + 1];
            }
            yPred[i] = sum;
        }

        return new Vector(yPred);
    }

    public Vector getWeights() {
        if (!fitted) {
            throw new IllegalStateException("Not fitted");
        }
        return weights;
    }

    public double getIntercept() {
        return getWeights().get(0);
    }

    public Vector getCoefficients() {
        Vector w = getWeights();
        double[] c = new double[w.size() - 1];
        for (int i = 1; i < w.size(); i++) {
            c[i - 1] = w.get(i);
        }
        return new Vector(c);
    }
}
