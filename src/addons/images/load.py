# -*- coding: utf-8
"""
Set of function for read and decode image.
"""
from typing import Tuple

import tensorflow as tf

from src.addons.images.mark import random_mark


def load_image(image_path: str) -> tf.Tensor:
    """
    Load an image as tensor.

    Parameters
    ----------
    image_path : str
        Path of image.

    Returns
    ------
    tf.Tensor
        Image as tensor.
    """
    image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image = tf.cast(tf.image.resize(image, (128, 128)), tf.uint8)
    return image


def get_image_and_mark(image_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load an image as tensor and generate a mark image.

    Parameters
    ----------
    image_path : str
        Path of image.

    Returns
    ------
    Tuple[tf.Tensor, tf.Tensor]
        Image as tensor and corresponding mark.
    """
    return load_image(image_path), random_mark(batch_size=1, shape=8)[0]
