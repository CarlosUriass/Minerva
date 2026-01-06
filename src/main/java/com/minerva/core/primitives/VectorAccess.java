package com.minerva.core.primitives;

/**
 * Provides privileged access to Vector internals for performance-critical
 * operations.
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
 * @see Vector
 */
public final class VectorAccess {

    // Prevent instantiation
    private VectorAccess() {
        throw new AssertionError("VectorAccess cannot be instantiated");
    }

    /**
     * Returns direct access to the internal data array of a Vector.
     * <p>
     * <strong>WARNING:</strong> This returns the actual internal array, not a copy.
     * Modifications will corrupt the Vector. Use only for read-only operations.
     * 
     * @param v the vector to access
     * @return direct reference to the internal data array (zero-copy)
     */
    public static double[] raw(Vector v) {
        return v.data;
    }
}
