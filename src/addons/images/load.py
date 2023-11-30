# -*- coding: utf-8
"""
Set of function for read and decode image.
"""
import tensorflow as tf


def load_image(image_path: str, height: int, width: int) -> tf.Tensor:
    """
    Load an image as tensor.

    Parameters
    ----------
    image_path : str
        path of image
    height : int
        New height of image
    width : int
        New width of images

    Returns
    ------
    tf.Tensor
        Image as tensor
    """
    image = tf.image.decode_image(tf.io.read_file(image_path))
    if tf.shape(image)[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    image = tf.image.resize(image, [height, width])
    return image
