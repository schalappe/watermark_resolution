# -*- coding: utf-8
"""
    Custom metrics
"""
import tensorflow as tf


def peak_signal_noise_ratio(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the peak signal-to-noise ratio (PSNR)
    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth values
    y_pred: tf.Tensor
        The predicted values

    Returns
    -------
        Peak signal-to-noise ratio value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=[-3, -2, -1])
    return tf.math.subtract(
        20.0 * tf.math.log(255.0) / tf.math.log(10.0),
        10.0 / tf.math.log(10.0) * tf.math.log(mse),
    )


def bit_error_ratio(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the bit error ratio (BER)
    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth values
    y_pred: tf.Tensor
        The predicted values

    Returns
    -------
        bit error ration
    """
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    return tf.reduce_mean(
        tf.cast(tf.math.equal(y_true, y_pred), tf.float32), axis=[-3, -2, -1]
    )


class PSNR(tf.keras.metrics.MeanMetricWrapper):
    """
    Calculates the peak signal-to-noise ratio
    """

    def __init__(self, name="peak_signal_to_noise_ratio", dtype=None):
        super().__init__(peak_signal_noise_ratio, name, dtype=dtype)


class BER(tf.keras.metrics.MeanMetricWrapper):
    """
    Calculates the bit error ratio
    """

    def __init__(self, name="bit_error_ratio", dtype=None):
        super().__init__(bit_error_ratio, name, dtype=dtype)
