package com.minerva.core.linalg;

/**
 * Algorithm selection for matrix inversion and solving linear systems.
 *
 * <p>
 * Choose based on matrix properties:
 * <ul>
 * <li>{@link #LU} - General purpose, works for any invertible matrix</li>
 * <li>{@link #CHOLESKY} - Fastest, but only for symmetric positive definite
 * matrices</li>
 * <li>{@link #QR} - Most stable, good for ill-conditioned matrices</li>
 * </ul>
 */
public enum InversionMethod {

    /**
     * LU decomposition with partial pivoting.
     * <p>
     * Complexity: O(2n³/3)
     * <p>
     * Use for: General invertible matrices
     */
    LU,

    /**
     * Cholesky decomposition (LLᵀ).
     * <p>
     * Complexity: O(n³/3) - about 2x faster than LU
     * <p>
     * Use for: Symmetric positive definite matrices (covariance, Gram matrices)
     * <p>
     * Throws: {@link com.minerva.core.exception.NonPositiveDefiniteException} if
     * not SPD
     */
    CHOLESKY,

    /**
     * QR decomposition with Householder reflectors.
     * <p>
     * Complexity: O(4n³/3)
     * <p>
     * Use for: Ill-conditioned matrices, least squares problems
     */
    QR
}
