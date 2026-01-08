package com.minerva.metrics.RegressionMetrics;

import com.minerva.core.primitives.Vector;
import com.minerva.core.primitives.VectorAccess;
import com.minerva.metrics.RegressionMetrics.api.IRegressionMetrics;

public class RegressionMetrics implements IRegressionMetrics {

    @Override
    public double MAE(Vector actual, Vector predicted) {
        double[] yActual = VectorAccess.raw(actual);
        double[] yPredicted = VectorAccess.raw(predicted);

        int n = yActual.length;

        if (n != yPredicted.length) {
            throw new IllegalArgumentException("Actual and predicted data must have the same length");
        }

        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.abs(yActual[i] - yPredicted[i]);
        }
        return sum / n;
    }

    @Override
    public double MSE(Vector actual, Vector predicted) {
        double[] yActual = VectorAccess.raw(actual);
        double[] yPredicted = VectorAccess.raw(predicted);

        int n = yActual.length;
        if (n != yPredicted.length) {
            throw new IllegalArgumentException("Length mismatch");
        }

        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = yActual[i] - yPredicted[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    @Override
    public double RMSE(Vector actual, Vector predicted) {
        return Math.sqrt(MSE(actual, predicted));
    }

    @Override
    public double R2(Vector actual, Vector predicted) {
        double[] yActual = VectorAccess.raw(actual);
        double[] yPredicted = VectorAccess.raw(predicted);

        int n = yActual.length;
        if (n != yPredicted.length) {
            throw new IllegalArgumentException("Length mismatch");
        }

        // mean of yActual
        double mean = 0.0;
        for (int i = 0; i < n; i++) {
            mean += yActual[i];
        }
        mean /= n;

        // SSE y SST
        double sse = 0.0;
        double sst = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = yActual[i] - yPredicted[i];
            sse += diff * diff;

            double dev = yActual[i] - mean;
            sst += dev * dev;
        }

        if (sst == 0.0) {
            return 0.0;
        }

        return 1.0 - sse / sst;
    }

    @Override
    public double R2adj(
            Vector actual,
            Vector predicted,
            int numFeatures) {
        double[] y = VectorAccess.raw(actual);
        double[] yHat = VectorAccess.raw(predicted);

        int n = y.length;
        if (n != yHat.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }

        if (n <= numFeatures + 1) {
            throw new IllegalArgumentException("Not enough samples");
        }

        // mean
        double mean = 0.0;
        for (double v : y)
            mean += v;
        mean /= n;

        double sse = 0.0;
        double sst = 0.0;

        for (int i = 0; i < n; i++) {
            double d = y[i] - yHat[i];
            sse += d * d;

            double dev = y[i] - mean;
            sst += dev * dev;
        }

        if (sst == 0.0)
            return 0.0;

        double r2 = 1.0 - (sse / sst);

        return 1.0 - (1.0 - r2) * (n - 1.0) / (n - numFeatures - 1.0);
    }

    @Override
    public double RMSLE(Vector actual, Vector predicted) {
        double[] yActual = VectorAccess.raw(actual);
        double[] yPredicted = VectorAccess.raw(predicted);

        int n = yActual.length;

        if (n != yPredicted.length) {
            throw new IllegalArgumentException("Actual and predicted data must have the same length");
        }

        double sum = 0;
        for (int i = 0; i < n; i++) {
            double logActual = Math.log1p(yActual[i]);
            double logPredicted = Math.log1p(yPredicted[i]);
            sum += Math.pow(logActual - logPredicted, 2);
        }
        return Math.sqrt(sum / n);
    }

    @Override
    public double MAPE(Vector actual, Vector predicted) {
        double[] yActual = VectorAccess.raw(actual);
        double[] yPredicted = VectorAccess.raw(predicted);

        int n = yActual.length;

        if (n != yPredicted.length) {
            throw new IllegalArgumentException("Actual and predicted data must have the same length");
        }

        double sum = 0;
        for (int i = 0; i < n; i++) {
            if (yActual[i] != 0) {
                sum += Math.abs((yActual[i] - yPredicted[i]) / yActual[i]);
            }
        }
        return (sum / n) * 100;
    }
}
