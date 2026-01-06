package com.minerva.core.stats.impl;

import com.minerva.core.api.Tensor;
import com.minerva.core.stats.api.TensorStats;

/**
 * Abstract base class providing common statistical implementations.
 * Subclasses can override for optimized implementations.
 */
public abstract class TensorStatsBase implements TensorStats {

    protected final Tensor tensor;

    protected TensorStatsBase(Tensor tensor) {
        this.tensor = tensor;
    }

    @Override
    public double avg() {
        // TODO: implement using tensor.get() and tensor.size()
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public double std() {
        // TODO: implement std = sqrt(variance)
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public double max() {
        // TODO: implement by iterating through tensor
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public double min() {
        // TODO: implement by iterating through tensor
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public double sum() {
        // TODO: implement by summing all elements
        throw new UnsupportedOperationException("Not implemented yet");
    }

    @Override
    public double variance() {
        // TODO: implement variance calculation
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
