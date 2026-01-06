package com.minerva.core.exception;

/**
 * Exception thrown when tensor operations encounter incompatible dimensions.
 */
public class DimensionMismatchException extends RuntimeException {

    public DimensionMismatchException() {
        super();
    }

    public DimensionMismatchException(String message) {
        super(message);
    }

    public DimensionMismatchException(String message, Throwable cause) {
        super(message, cause);
    }
}
