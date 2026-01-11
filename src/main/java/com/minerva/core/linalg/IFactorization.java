package com.minerva.core.linalg;

import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.Vector;

/**
 * Interface for matrix factorizations that support solving linear systems.
 *
 * <p>
 * Factorizations decompose a matrix A into products of simpler matrices,
 * enabling efficient solutions to Ax = b without computing A⁻¹ explicitly.
 *
 * <p>
 * Usage pattern:
 * 
 * <pre>{@code
 * IFactorization lu = new LUFactorization(A);
 * Vector x = lu.solve(b); // Solve Ax = b
 * Matrix inv = lu.inverse(); // Compute A⁻¹ (use sparingly)
 * }</pre>
 *
 * <p>
 * <strong>Performance tip:</strong> Reuse factorization when solving
 * multiple systems with the same matrix A.
 */
public interface IFactorization {

    /**
     * Solves the linear system Ax = b for x.
     *
     * @param b the right-hand side vector
     * @return the solution vector x
     * @throws IllegalArgumentException if b has wrong dimension
     */
    Vector solve(Vector b);

    /**
     * Solves the linear system AX = B for X (multiple right-hand sides).
     *
     * @param B the right-hand side matrix (each column is a separate system)
     * @return the solution matrix X
     * @throws IllegalArgumentException if B has wrong dimensions
     */
    Matrix solve(Matrix B);

    /**
     * Computes the matrix inverse A⁻¹.
     *
     * <p>
     * <strong>Note:</strong> Prefer {@link #solve(Vector)} when possible.
     * Computing the inverse is ~2x slower and less numerically stable.
     *
     * @return the inverse matrix
     */
    Matrix inverse();

    /**
     * Returns the dimension of the factorized matrix (n for n×n matrix).
     *
     * @return matrix dimension
     */
    int size();
}
