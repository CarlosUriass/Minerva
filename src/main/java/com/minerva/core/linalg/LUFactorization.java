package com.minerva.core.linalg;

import com.minerva.core.exception.SingularMatrixException;
import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.MatrixAccess;
import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;

/**
 * LU Decomposition with partial pivoting (Doolittle algorithm).
 *
 * <p>
 * Decomposes matrix A into PA = LU where:
 * <ul>
 * <li>P is a permutation matrix (stored as pivot indices)</li>
 * <li>L is lower triangular with unit diagonal</li>
 * <li>U is upper triangular</li>
 * </ul>
 *
 * <p>
 * Storage: L and U are stored compactly in a single array.
 * L occupies the strict lower triangle, U occupies the upper triangle
 * (including diagonal).
 */
public final class LUFactorization implements IFactorization {

    private static final double SINGULARITY_THRESHOLD = 1e-12;

    private final int n;
    private final double[] lu; // Compact L\U storage (row-major)
    private final int[] pivot; // Permutation indices
    private final int pivotSign; // +1 or -1 (for determinant)

    /**
     * Computes the LU decomposition of the given square matrix.
     *
     * @param A the matrix to decompose
     * @throws IllegalArgumentException if matrix is not square
     * @throws SingularMatrixException  if matrix is singular
     */
    public LUFactorization(Matrix A) {
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("Matrix must be square: " + A.rows() + "x" + A.cols());
        }

        this.n = A.rows();
        this.lu = MatrixAccess.raw(A).clone();
        this.pivot = new int[n];

        // Initialize pivot indices
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }

        int sign = 1;

        // Doolittle LU with partial pivoting
        for (int k = 0; k < n; k++) {
            // Find pivot: maximum absolute value in column k, rows k..n-1
            int maxRow = k;
            double maxVal = Math.abs(lu[k * n + k]);

            for (int i = k + 1; i < n; i++) {
                double val = Math.abs(lu[i * n + k]);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = i;
                }
            }

            // Check for singularity
            if (maxVal < SINGULARITY_THRESHOLD) {
                throw new SingularMatrixException("Matrix is singular or nearly singular (pivot = " + maxVal + ")");
            }

            // Swap rows if needed
            if (maxRow != k) {
                // Swap in lu array
                int rowK = k * n;
                int rowMax = maxRow * n;
                for (int j = 0; j < n; j++) {
                    double tmp = lu[rowK + j];
                    lu[rowK + j] = lu[rowMax + j];
                    lu[rowMax + j] = tmp;
                }
                // Swap pivot indices
                int tmp = pivot[k];
                pivot[k] = pivot[maxRow];
                pivot[maxRow] = tmp;
                sign = -sign;
            }

            // Elimination
            double pivot_kk = lu[k * n + k];
            for (int i = k + 1; i < n; i++) {
                int rowI = i * n;
                double factor = lu[rowI + k] / pivot_kk;
                lu[rowI + k] = factor; // Store L factor

                for (int j = k + 1; j < n; j++) {
                    lu[rowI + j] -= factor * lu[k * n + j];
                }
            }
        }

        this.pivotSign = sign;
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

        // Apply permutation: x = P * b
        double[] pb = new double[n];
        for (int i = 0; i < n; i++) {
            pb[i] = x[pivot[i]];
        }

        // Forward substitution: Ly = Pb
        for (int i = 1; i < n; i++) {
            double sum = pb[i];
            int rowI = i * n;
            for (int j = 0; j < i; j++) {
                sum -= lu[rowI + j] * pb[j];
            }
            pb[i] = sum;
        }

        // Back substitution: Ux = y
        for (int i = n - 1; i >= 0; i--) {
            double sum = pb[i];
            int rowI = i * n;
            for (int j = i + 1; j < n; j++) {
                sum -= lu[rowI + j] * pb[j];
            }
            pb[i] = sum / lu[rowI + i];
        }

        return new Vector(pb);
    }

    @Override
    public Matrix solve(Matrix B) {
        if (B.rows() != n) {
            throw new IllegalArgumentException("Matrix rows " + B.rows() + " != size " + n);
        }

        int m = B.cols();
        double[] bData = MatrixAccess.raw(B);
        double[] result = new double[n * m];

        // Solve column by column
        for (int col = 0; col < m; col++) {
            // Extract column with permutation applied
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = bData[pivot[i] * m + col];
            }

            // Forward substitution: Ly = Pb
            for (int i = 1; i < n; i++) {
                double sum = x[i];
                int rowI = i * n;
                for (int j = 0; j < i; j++) {
                    sum -= lu[rowI + j] * x[j];
                }
                x[i] = sum;
            }

            // Back substitution: Ux = y
            for (int i = n - 1; i >= 0; i--) {
                double sum = x[i];
                int rowI = i * n;
                for (int j = i + 1; j < n; j++) {
                    sum -= lu[rowI + j] * x[j];
                }
                x[i] = sum / lu[rowI + i];
            }

            // Store column in result (row-major)
            for (int i = 0; i < n; i++) {
                result[i * m + col] = x[i];
            }
        }

        // Use factory method via identity and solve approach
        // Build result matrix directly
        double[][] resultArr = new double[n][m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(result, i * m, resultArr[i], 0, m);
        }
        return new Matrix(resultArr);
    }

    @Override
    public Matrix inverse() {
        return solve(Matrix.identity(n));
    }

    /**
     * Returns the determinant of the original matrix.
     *
     * @return det(A)
     */
    public double determinant() {
        double det = pivotSign;
        for (int i = 0; i < n; i++) {
            det *= lu[i * n + i];
        }
        return det;
    }
}
