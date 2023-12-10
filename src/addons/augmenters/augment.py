# -*- coding: utf-8 -*-
"""
Set of functions for data augmentation.
"""
import tensorflow as tf
from random import choices
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
    identity,
)
from typing import Callable


@tf.py_function(Tout=tf.uint8)
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

    return tf.cast(images, tf.uint8)


attacks = {
    "crop": random_crop,
    "dropout": random_dropout,
    "identity": identity,
    "rotation": random_rotate,
    "salt_pepper": random_salt_pepper,
    "median_blur": random_median_blur,
    "average_blur": random_average_blur,
    "gaussian_blur": random_gaussian_blur,
    "image_quality": random_jpeg_quality,
    "gaussian_noise": random_gaussian_noise,
}

attack_probs = {
    "crop": 1,
    "dropout": 1,
    "identity": 1,
    "rotation": 1,
    "salt_pepper": 1,
    "median_blur": 1,
    "average_blur": 2,
    "gaussian_blur": 2,
    "image_quality": 1,
    "gaussian_noise": 1,
}


def random_attacks() -> Callable:
    """
    Randomly choose an attack function.

    Returns
    -------
    Callable
        Chosen attack function.
    """
    attack = choices(list(attack_probs.keys()), weights=list(attack_probs.values()), k=1)
    return attacks[attack]
