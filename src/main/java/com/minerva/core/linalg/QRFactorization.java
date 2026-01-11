package com.minerva.core.linalg;

import com.minerva.core.exception.SingularMatrixException;
import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;

/**
 * QR Decomposition using Householder reflectors.
 *
 * <p>
 * Decomposes matrix A into A = QR where:
 * <ul>
 * <li>Q is orthogonal (Qᵀ = Q⁻¹)</li>
 * <li>R is upper triangular</li>
 * </ul>
 *
 * <p>
 * Advantages:
 * <ul>
 * <li>Most numerically stable decomposition</li>
 * <li>Works for rectangular matrices (least squares)</li>
 * <li>Better for ill-conditioned matrices</li>
 * </ul>
 *
 * <p>
 * Disadvantages:
 * <ul>
 * <li>~2x slower than LU for inversion</li>
 * </ul>
 */
public final class QRFactorization implements IFactorization {

    private static final double TOLERANCE = 1e-12;

    private final int m; // Rows
    private final int n; // Columns
    private final double[] qr; // Compact Householder storage
    private final double[] rDiag; // Diagonal of R

    /**
     * Computes the QR decomposition of the given matrix.
     *
     * @param A the matrix to decompose (m x n, m >= n for least squares)
     * @throws IllegalArgumentException if m < n
     */
    public QRFactorization(Matrix A) {
        this.m = A.rows();
        this.n = A.cols();

        if (m < n) {
            throw new IllegalArgumentException("QR requires m >= n, got " + m + "x" + n);
        }

        this.qr = MatrixAccess.raw(A).clone();
        this.rDiag = new double[n];

        // Householder QR
        for (int k = 0; k < n; k++) {
            // Compute norm of column k below diagonal
            double norm = 0.0;
            for (int i = k; i < m; i++) {
                norm = hypot(norm, qr[i * n + k]);
            }

            if (norm > TOLERANCE) {
                // Form k-th Householder vector
                if (qr[k * n + k] < 0) {
                    norm = -norm;
                }
                for (int i = k; i < m; i++) {
                    qr[i * n + k] /= norm;
                }
                qr[k * n + k] += 1.0;

                // Apply transformation to remaining columns
                for (int j = k + 1; j < n; j++) {
                    double s = 0.0;
                    for (int i = k; i < m; i++) {
                        s += qr[i * n + k] * qr[i * n + j];
                    }
                    s = -s / qr[k * n + k];
                    for (int i = k; i < m; i++) {
                        qr[i * n + j] += s * qr[i * n + k];
                    }
                }
            }
            rDiag[k] = -norm;
        }
    }

    /**
     * Computes sqrt(a² + b²) without overflow/underflow.
     */
    private static double hypot(double a, double b) {
        if (Math.abs(a) > Math.abs(b)) {
            double r = b / a;
            return Math.abs(a) * Math.sqrt(1 + r * r);
        } else if (b != 0) {
            double r = a / b;
            return Math.abs(b) * Math.sqrt(1 + r * r);
        } else {
            return 0.0;
        }
    }

    @Override
    public int size() {
        if (m != n) {
            throw new IllegalStateException("size() only valid for square matrices");
        }
        return n;
    }

    /**
     * Returns true if R is full rank (no zero diagonal elements).
     */
    public boolean isFullRank() {
        for (int j = 0; j < n; j++) {
            if (Math.abs(rDiag[j]) < TOLERANCE) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Vector solve(Vector b) {
        if (b.size() != m) {
            throw new IllegalArgumentException("Vector size " + b.size() + " != matrix rows " + m);
        }
        if (!isFullRank()) {
            throw new SingularMatrixException("Matrix is rank deficient");
        }

        double[] x = VectorAccess.raw(b).clone();

        // Compute Qᵀb
        for (int k = 0; k < n; k++) {
            double s = 0.0;
            for (int i = k; i < m; i++) {
                s += qr[i * n + k] * x[i];
            }
            s = -s / qr[k * n + k];
            for (int i = k; i < m; i++) {
                x[i] += s * qr[i * n + k];
            }
        }

        // Solve Rx = Qᵀb by back substitution
        for (int k = n - 1; k >= 0; k--) {
            x[k] /= rDiag[k];
            for (int i = 0; i < k; i++) {
                x[i] -= x[k] * qr[i * n + k];
            }
        }

        // Truncate to n elements
        double[] result = new double[n];
        System.arraycopy(x, 0, result, 0, n);
        return new Vector(result);
    }

    @Override
    public Matrix solve(Matrix B) {
        if (B.rows() != m) {
            throw new IllegalArgumentException("Matrix rows " + B.rows() + " != " + m);
        }
        if (!isFullRank()) {
            throw new SingularMatrixException("Matrix is rank deficient");
        }

        int cols = B.cols();
        double[] bData = MatrixAccess.raw(B);
        double[][] result = new double[n][cols];

        for (int col = 0; col < cols; col++) {
            // Extract column
            double[] x = new double[m];
            for (int i = 0; i < m; i++) {
                x[i] = bData[i * cols + col];
            }

            // Compute Qᵀb
            for (int k = 0; k < n; k++) {
                double s = 0.0;
                for (int i = k; i < m; i++) {
                    s += qr[i * n + k] * x[i];
                }
                s = -s / qr[k * n + k];
                for (int i = k; i < m; i++) {
                    x[i] += s * qr[i * n + k];
                }
            }

            // Back substitution
            for (int k = n - 1; k >= 0; k--) {
                x[k] /= rDiag[k];
                for (int i = 0; i < k; i++) {
                    x[i] -= x[k] * qr[i * n + k];
                }
            }

            // Store column
            for (int i = 0; i < n; i++) {
                result[i][col] = x[i];
            }
        }

        return new Matrix(result);
    }

    @Override
    public Matrix inverse() {
        if (m != n) {
            throw new IllegalStateException("inverse() only valid for square matrices");
        }
        return solve(Matrix.identity(n));
    }
}
