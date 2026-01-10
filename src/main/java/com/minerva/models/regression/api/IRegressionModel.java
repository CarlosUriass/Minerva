package com.minerva.models.regression.api;

import com.minerva.core.primitives.Matrix;
import com.minerva.core.primitives.Vector;

public interface IRegressionModel {

    /**
     * Train the model.
     *
     * @param X Design matrix (n_samples × n_features)
     * @param y Target values (n_samples)
     */
    void fit(Matrix X, Vector y);

    /**
     * Predict target values.
     *
     * @param X Design matrix (n_samples × n_features)
     * @return Predicted values (n_samples)
     */
    Vector predict(Matrix X);
}
