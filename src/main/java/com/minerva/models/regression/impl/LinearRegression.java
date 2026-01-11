package com.minerva.models.regression.impl;

import com.minerva.core.linalg.QRFactorization;
import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;
import com.minerva.models.regression.api.IRegressionModel;

/**
 * Ordinary Least Squares Linear Regression using QR Decomposition.
 *
 * <p>
 * Solves:
 * 
 * <pre>
 *   min ||X w - y||²
 * </pre>
 *
 * with implicit intercept (bias term).
 *
 * <p>
 * Uses {@link QRFactorization} for numerical stability.
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

        // Build augmented design matrix X̃ = [1 | X]
        double[][] augmented = new double[n][k];
        double[] xRaw = MatrixAccess.raw(X);

        for (int i = 0; i < n; i++) {
            augmented[i][0] = 1.0; // intercept column
            System.arraycopy(xRaw, i * p, augmented[i], 1, p);
        }

        Matrix XAug = new Matrix(augmented);

        // Solve X̃ᵀX̃ w = X̃ᵀy using QR decomposition on X̃
        // QR handles this directly via least squares
        QRFactorization qr = new QRFactorization(XAug);
        this.weights = qr.solve(y);
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

    /**
     * Returns all weights including intercept: [intercept, w1, w2, ...]
     */
    public Vector getWeights() {
        if (!fitted) {
            throw new IllegalStateException("Not fitted");
        }
        return weights;
    }

    /**
     * Returns the intercept (bias) term.
     */
    public double getIntercept() {
        return getWeights().get(0);
    }

    /**
     * Returns coefficients without intercept: [w1, w2, ...]
     */
    public Vector getCoefficients() {
        Vector w = getWeights();
        double[] c = new double[w.size() - 1];
        for (int i = 1; i < w.size(); i++) {
            c[i - 1] = w.get(i);
        }
        return new Vector(c);
    }
}
