package com.minerva.core.api;

/**
 * Base interface for all tensor-like structures.
 * Provides common operations for n-dimensional numerical data.
 */
public interface Tensor {

    /**
     * Returns the shape of the tensor as an array of dimensions.
     * 
     * @return array where each element represents the size of that dimension
     */
    int[] shape();

    /**
     * Returns the total number of elements in the tensor.
     * 
     * @return total element count
     */
    int size();

    /**
     * Gets the value at the specified indices.
     * 
     * @param indices position in each dimension
     * @return the value at the specified position
     */
    double get(int... indices);

    /**
     * Sets the value at the specified indices.
     * 
     * @param value   the value to set
     * @param indices position in each dimension
     */
    void set(double value, int... indices);
}
