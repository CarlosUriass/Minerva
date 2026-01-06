package com.minerva.core.api;

import com.minerva.core.primitives.Vector;

/**
 * Defines common vector operations for linear algebra.
 * <p>
 * This interface provides methods for vector arithmetic, dot products,
 * scaling, and norm calculations.
 */
public interface VectorOps {

    /**
     * Computes the dot product (inner product) of this vector with another.
     * <p>
     * The dot product is defined as: Σ(a[i] * b[i])
     * 
     * @param other the vector to compute the dot product with
     * @return the scalar dot product
     * @throws IllegalArgumentException if vectors have different sizes
     */
    double dot(Vector other);

    /**
     * Adds this vector to another vector element-wise.
     * 
     * @param other the vector to add
     * @return a new vector containing the sum
     * @throws IllegalArgumentException if vectors have different sizes
     */
    Vector add(Vector other);

    /**
     * Subtracts another vector from this vector element-wise.
     * 
     * @param other the vector to subtract
     * @return a new vector containing the difference
     * @throws IllegalArgumentException if vectors have different sizes
     */
    Vector subtract(Vector other);

    /**
     * Multiplies this vector by a scalar value.
     * 
     * @param scalar the value to multiply each element by
     * @return a new vector with each element scaled
     */
    Vector scale(double scalar);

    /**
     * Computes the Euclidean norm (L2 norm, magnitude) of this vector.
     * <p>
     * The norm is defined as: √(Σ(x[i]²))
     * 
     * @return the norm of the vector
     */
    double norm();
}
