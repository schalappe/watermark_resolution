# -*- coding: utf-8
"""
Custom metrics
"""
import tensorflow as tf
import keras
from keras.losses import MeanSquaredError


def peak_signal_noise_ratio(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the peak signal-to-noise ratio (PSNR).

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    tf.Tensor
        Peak signal-to-noise ratio.
    """
    mse = tf.math.reduce_mean(tf.math.square(y_true - y_pred), axis=[-3, -2, -1])
    signal_noise = 10 * (tf.math.log((255**2) / mse) / tf.math.log(10.0))
    return tf.math.reduce_mean(signal_noise)


def bit_error_ratio(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the bit error ratio (BER).

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values.
    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    tf.Tensor
        bit error ratio.
    """
    bit_error = 1 - tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
    area = tf.cast(tf.size(y_true[0]), tf.float32)
    ber = 100 * (tf.math.reduce_sum(bit_error, axis=[-3, -2, -1]) / area)
    return tf.math.reduce_mean(ber)


class PeakSignalNoiseRatio(keras.metrics.MeanMetricWrapper):
    """
    Calculates the peak signal-to-noise ratio.
    """

    def __init__(self, name="peak_signal_to_noise_ratio", dtype=None):
        super().__init__(peak_signal_noise_ratio, name, dtype=dtype)


class BitErrorRatio(keras.metrics.MeanMetricWrapper):
    """
    Calculates the bit error ratio.
    """

    def __init__(self, name="bit_error_ratio", dtype=None):
        super().__init__(bit_error_ratio, name, dtype=dtype)
