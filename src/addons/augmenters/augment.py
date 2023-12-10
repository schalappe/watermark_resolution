# -*- coding: utf-8 -*-
"""
Set of functions for data augmentation.
"""
import tensorflow as tf
from src.addons.augmenters.base import (
    random_hue,
    random_flip,
    random_saturation,
    random_brightness,
    random_contrast,
    random_rotate,
    random_dropout,
    random_average_blur,
    random_gaussian_blur,
    random_median_blur,
    random_salt_pepper,
    random_gaussian_noise,
    random_jpeg_quality,
    random_crop,
)


@tf.py_function(Tout=tf.float32)
def augment(images: tf.Tensor) -> tf.Tensor:
    """
    Take an image and apply random data augmentation.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Augmented image.
    """
    augmentation = [random_flip, random_hue, random_saturation, random_brightness, random_contrast]

    for func in augmentation:
        if tf.random.uniform([], 0, 1) > 0.25:
            images = func(images)

    return images


attacks = {
    "rotation": random_rotate,
    "dropout": random_dropout,
    "crop": random_crop,
    "gaussian_blur": random_gaussian_blur,
    "average_blur": random_average_blur,
    "median_blur": random_median_blur,
    "gaussian_noise": random_gaussian_noise,
    "salt_pepper": random_salt_pepper,
    "image_quality": random_jpeg_quality,
}
