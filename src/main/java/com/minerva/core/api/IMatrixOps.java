package com.minerva.core.api;

import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.Vector;

/**
 * Interface defining matrix-specific operations.
 */
public interface IMatrixOps {

    /**
     * Multiplies this matrix with another matrix.
     * 
     * @param other the matrix to multiply with
     * @return the resulting matrix
     */
    Matrix multiply(Matrix other);

    /**
     * Multiplies this matrix with a vector.
     * 
     * @param vector the vector to multiply with
     * @return the resulting vector
     */
    Vector multiply(Vector vector);

    /**
     * Adds another matrix element-wise.
     * 
     * @param other the matrix to add
     * @return a new matrix with the result
     */
    Matrix add(Matrix other);

    /**
     * Subtracts another matrix element-wise.
     * 
     * @param other the matrix to subtract
     * @return a new matrix with the result
     */
    Matrix subtract(Matrix other);

    /**
     * Computes the transpose of this matrix.
     * 
     * @return the transposed matrix
     */
    Matrix transpose();

    /**
     * Returns the number of rows.
     * 
     * @return row count
     */
    int rows();

    /**
     * Returns the number of columns.
     * 
     * @return column count
     */
    int cols();
}
