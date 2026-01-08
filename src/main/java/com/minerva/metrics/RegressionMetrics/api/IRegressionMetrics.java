package com.minerva.metrics.RegressionMetrics.api;

import com.minerva.core.primitives.Vector;

public interface IRegressionMetrics {

    /**
     * Mean Absolute Error
     * 
     * @return
     */
    double MAE(Vector actual, Vector predicted);

    /**
     * Mean Squared Error
     * 
     * @return
     */
    double MSE(Vector actual, Vector predicted);

    double RMSE(Vector actual, Vector predicted);

    /**
     * R-squared
     * 
     * @return
     */
    double R2(Vector actual, Vector predicted);

    double R2adj(Vector actual, Vector predicted, int numFeatures);

    double RMSLE(Vector actual, Vector predicted);

    double MAPE(Vector actual, Vector predicted);

}
