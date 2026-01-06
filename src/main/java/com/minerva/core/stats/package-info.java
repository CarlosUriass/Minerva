/**
 * Statistical operations for tensor structures.
 * 
 * <p>
 * This package provides statistical analysis capabilities for Vectors,
 * Matrices,
 * and other tensor-like structures. The architecture follows a composition
 * pattern
 * where statistics are computed by specialized classes rather than being
 * embedded
 * in the primitive classes themselves.
 * 
 * <h2>Usage Example:</h2>
 * 
 * <pre>
 * Vector v = new Vector(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
 * VectorStats stats = new VectorStats(v);
 * 
 * double mean = stats.mean();
 * double std = stats.std();
 * double max = stats.max();
 * </pre>
 * 
 * <h2>Package Structure:</h2>
 * <ul>
 * <li>{@code api/} - Interfaces defining statistical contracts</li>
 * <li>{@code impl/} - Concrete implementations for different tensor types</li>
 * </ul>
 */
package com.minerva.core.stats;
