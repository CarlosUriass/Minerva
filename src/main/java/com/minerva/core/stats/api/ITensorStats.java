package com.minerva.core.stats.api;

/**
 * Generic interface for statistical operations on tensor-like structures.
 * This interface can be implemented for any Tensor (Vector, Matrix, etc.)
 */
public interface ITensorStats {

    /**
     * Computes the arithmetic average.
     * 
     * @return the average value
     */
    double avg();

    /**
     * Finds the median in the element
     * 
     * @return the median value
     */
    double median();

    /**
     * Finds the mode in the element
     * 
     * @return the mode value
     */
    double mode();

    /**
     * Computes the standard deviation of all elements.
     * 
     * @return the standard deviation
     */
    double std();

    /**
     * Finds the maximum value.
     * 
     * @return the maximum element
     */
    double max();

    /**
     * Finds the minimum value.
     * 
     * @return the minimum element
     */
    double min();

    /**
     * Computes the sum of all elements.
     * 
     * @return the sum
     */
    double sum();

    /**
     * Computes the variance of all elements.
     * 
     * @return the variance
     */
    double variance();

    /**
     * Computes the percentile of all elements.
     * 
     * @return the percentile
     */
    double percentile(double percentile);
}
