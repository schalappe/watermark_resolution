# -*- coding: utf-8
"""
Base preprocessor
"""
import tensorflow as tf


@tf.function
def load_image(image_path: str) -> tf.Tensor:
    """
    Load an image

    Parameters
    ----------
    image_path: str
        path of image

    Returns
    ------
    tf.Tensor: Image as tensor

    """
    # read the image from disk, decode it
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)

    # expands if necessary
    if len(tf.shape(image)) < 3:
        image = tf.expand_dims(image, axis=-1)

    # return the image
    return image
