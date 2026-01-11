package com.minerva.core.exception;

/**
 * Thrown when Cholesky decomposition fails because the matrix
 * is not symmetric positive definite (SPD).
 *
 * A matrix A is SPD if:
 * - A is symmetric: A = Aᵀ
 * - All eigenvalues are positive
 * - For all non-zero vectors x: xᵀAx > 0
 */
public class NonPositiveDefiniteException extends ArithmeticException {

    public NonPositiveDefiniteException() {
        super("Matrix is not positive definite");
    }

    public NonPositiveDefiniteException(String message) {
        super(message);
    }
}
