package com.minerva.core.linalg;

import com.minerva.core.exception.NonPositiveDefiniteException;
import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;

/**
 * Cholesky Decomposition for symmetric positive definite (SPD) matrices.
 *
 * <p>
 * Decomposes matrix A into A = LLᵀ where L is lower triangular.
 *
 * <p>
 * Advantages over LU:
 * <ul>
 * <li>~2x faster (only processes lower triangle)</li>
 * <li>Numerically stable (no pivoting needed for SPD)</li>
 * <li>Half the storage</li>
 * </ul>
 *
 * <p>
 * Common use cases in ML:
 * <ul>
 * <li>Covariance matrices</li>
 * <li>Gram matrices (XᵀX)</li>
 * <li>Kernel matrices</li>
 * </ul>
 */
public final class CholeskyFactorization implements IFactorization {

    private static final double TOLERANCE = 1e-12;

    private final int n;
    private final double[] L; // Lower triangular (row-major, full storage)

    /**
     * Computes the Cholesky decomposition of a symmetric positive definite matrix.
     *
     * @param A the SPD matrix to decompose
     * @throws IllegalArgumentException     if matrix is not square
     * @throws NonPositiveDefiniteException if matrix is not positive definite
     */
    public CholeskyFactorization(Matrix A) {
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("Matrix must be square: " + A.rows() + "x" + A.cols());
        }

        this.n = A.rows();
        this.L = new double[n * n];
        double[] aData = MatrixAccess.raw(A);

        // Cholesky-Banachiewicz algorithm
        for (int i = 0; i < n; i++) {
            int rowI = i * n;

            for (int j = 0; j <= i; j++) {
                int rowJ = j * n;
                double sum = aData[rowI + j];

                for (int k = 0; k < j; k++) {
                    sum -= L[rowI + k] * L[rowJ + k];
                }

                if (i == j) {
                    if (sum <= TOLERANCE) {
                        throw new NonPositiveDefiniteException(
                                "Matrix is not positive definite (diagonal element " + i + " = " + sum + ")");
                    }
                    L[rowI + j] = Math.sqrt(sum);
                } else {
                    L[rowI + j] = sum / L[rowJ + j];
                }
            }
        }
    }

    @Override
    public int size() {
        return n;
    }

    @Override
    public Vector solve(Vector b) {
        if (b.size() != n) {
            throw new IllegalArgumentException("Vector size " + b.size() + " != matrix size " + n);
        }

        double[] x = VectorAccess.raw(b).clone();

        // Forward substitution: Ly = b
        for (int i = 0; i < n; i++) {
            double sum = x[i];
            int rowI = i * n;
            for (int j = 0; j < i; j++) {
                sum -= L[rowI + j] * x[j];
            }
            x[i] = sum / L[rowI + i];
        }

        // Back substitution: Lᵀx = y
        for (int i = n - 1; i >= 0; i--) {
            double sum = x[i];
            for (int j = i + 1; j < n; j++) {
                sum -= L[j * n + i] * x[j]; // Access Lᵀ
            }
            x[i] = sum / L[i * n + i];
        }

        return new Vector(x);
    }

    @Override
    public Matrix solve(Matrix B) {
        if (B.rows() != n) {
            throw new IllegalArgumentException("Matrix rows " + B.rows() + " != size " + n);
        }

        int m = B.cols();
        double[] bData = MatrixAccess.raw(B);
        double[][] result = new double[n][m];

        // Extract and solve each column
        for (int col = 0; col < m; col++) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = bData[i * m + col];
            }

            // Forward substitution: Ly = b
            for (int i = 0; i < n; i++) {
                double sum = x[i];
                int rowI = i * n;
                for (int j = 0; j < i; j++) {
                    sum -= L[rowI + j] * x[j];
                }
                x[i] = sum / L[rowI + i];
            }

            // Back substitution: Lᵀx = y
            for (int i = n - 1; i >= 0; i--) {
                double sum = x[i];
                for (int j = i + 1; j < n; j++) {
                    sum -= L[j * n + i] * x[j];
                }
                x[i] = sum / L[i * n + i];
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
        return solve(Matrix.identity(n));
    }

    /**
     * Returns the determinant of the original matrix.
     * For SPD matrices, det(A) = det(L)² = (∏ Lᵢᵢ)²
     *
     * @return det(A)
     */
    public double determinant() {
        double det = 1.0;
        for (int i = 0; i < n; i++) {
            det *= L[i * n + i];
        }
        return det * det;
    }
}
