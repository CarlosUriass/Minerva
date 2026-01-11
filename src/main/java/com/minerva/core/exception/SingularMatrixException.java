package com.minerva.core.exception;

/**
 * Thrown when a matrix operation fails because the matrix is singular
 * (determinant is zero or numerically negligible).
 *
 * Common causes:
 * - Matrix has linearly dependent rows/columns
 * - LU decomposition encounters zero pivot
 * - Matrix is rank-deficient
 */
public class SingularMatrixException extends ArithmeticException {

    public SingularMatrixException() {
        super("Matrix is singular");
    }

    public SingularMatrixException(String message) {
        super(message);
    }
}
