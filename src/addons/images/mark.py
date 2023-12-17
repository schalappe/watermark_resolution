# -*- coding: utf-8
"""
Image generate ops
"""
import tensorflow as tf


@tf.function
def random_mark(batch_size: int, shape: int) -> tf.Tensor:
    """
    Generate random binary image with uniform distribution.

    Parameters
    ----------
    batch_size: int
        Batch size.
    shape: int
        Shape of image.

    Returns
    -------
        Random binary image.
    """
    return tf.cast(
        tf.random.uniform(shape=(batch_size, shape, shape, 1), minval=0, maxval=2, dtype=tf.int32),
        tf.float32,
    )
