# -*- coding: utf-8
"""
Set of function for read and decode image.
"""
import tensorflow as tf


def load_image(image_path: str) -> tf.Tensor:
    """
    Load an image as tensor.

    Parameters
    ----------
    image_path : str
        path of image

    Returns
    ------
    tf.Tensor
        Image as tensor
    """
    image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (512, 512))
    return image
