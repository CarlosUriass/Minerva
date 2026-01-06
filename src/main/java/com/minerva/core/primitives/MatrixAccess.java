package com.minerva.core.primitives;

/**
 * Provides privileged access to Matrix internals for performance-critical
 * operations.
 * 
 * <p>
 * This class uses the "friend-like access" pattern common in Java numerical
 * libraries.
 * It allows trusted code (like statistics and algorithms) to access raw
 * internal data
 * without exposing public getters or breaking encapsulation.
 * 
 * <p>
 * <strong>WARNING:</strong> Methods in this class return direct references to
 * internal
 * arrays. Do NOT modify the returned arrays. This is for read-only,
 * performance-critical
 * operations only.
 * 
 * @see Matrix
 */
public final class MatrixAccess {

    // Prevent instantiation
    private MatrixAccess() {
        throw new AssertionError("MatrixAccess cannot be instantiated");
    }

    /**
     * Returns direct access to the internal data array of a Matrix.
     * <p>
     * <strong>WARNING:</strong> This returns the actual internal array, not a copy.
     * Modifications will corrupt the Matrix. Use only for read-only operations.
     * 
     * @param m the matrix to access
     * @return direct reference to the internal data array (zero-copy)
     */
    public static double[] raw(Matrix m) {
        return m.data;
    }
}
