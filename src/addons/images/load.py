# -*- coding: utf-8
"""
Set of function for read and decode image.
"""
import tensorflow as tf
from src.addons.images.mark import random_mark
from typing import Tuple


def load_image(image_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load an image as tensor.

    Parameters
    ----------
    image_path : str
        path of image

    Returns
    ------
    Tuple[tf.Tensor, tf.Tensor]
        Image as tensor and corresponding mark
    """
    image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image = tf.cast(tf.image.resize(image, (512, 512)), tf.float32)
    return image, random_mark(batch_size=1, shape=32)[0]
