# -*- coding: utf-8
"""
Useful function for training and testing
"""
import tensorflow as tf


def preprocess_input(inputs: tf.Tensor, mode: str) -> tf.Tensor:
    """
    Preprocesses a tensor encoding a batch of images (Normalization)
    Parameters
    ----------
    inputs: tf.Tensor
        Input tensor, 3D or 4D.

    mode: str
        One of "tf" or "torch".
            - tf: will scale pixels between -1 and 1,
            - torch: will scale pixels between 0 and 1

    Returns
    -------
        Preprocessed tensor.
    """
    inputs = tf.cast(inputs, dtype=tf.float32)
    if mode == "tf":
        inputs /= 127.5
        inputs -= 1.0
        return inputs

    if mode == "torch":
        inputs /= 255.0
        return inputs

    return inputs


def preprocess_output(outputs: tf.Tensor, mode: str) -> tf.Tensor:
    """
    Preprocesses a tensor encoding a batch of images (De-normalization)
    Parameters
    ----------
    outputs: tf.Tensor
        Input tensor, 3D or 4D.

    mode: str
        One of "tf" or "torch".
            - tf: will scale pixels between 0 and 255 from -1 and 1
            - torch: will scale pixels between 0 and 255 from 0 and 1

    Returns
    -------
        Preprocessed tensor.
    """
    if mode == "tf":
        outputs += 1.0
        outputs *= 127.5
        return outputs

    if mode == "torch":
        outputs *= 255.0
        return outputs

    return outputs
