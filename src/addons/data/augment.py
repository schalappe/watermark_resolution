# -*- coding: utf-8 -*-
"""
Set of functions for data augmentation.
"""
import tensorflow as tf
from random import choices
from src.addons.data.base import (
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
from typing import Tuple


@tf.numpy_function(Tout=[tf.float32, tf.float32])
def augment(images: tf.Tensor, marks: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Take an image and apply random data augmentation.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented an image.
    marks : tf.Tensor
        A tensor represented an mark.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        Augmented image and mark tensors.
    """
    augmentation = [random_flip, random_hue, random_saturation, random_brightness, random_contrast]

    for func in augmentation:
        if tf.random.uniform([], 0, 1) > 0.25:
            images = func(images)

    return tf.cast(images, tf.float32), marks


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


@tf.numpy_function(Tout=tf.float32)
def random_attacks(images: tf.Tensor) -> tf.Tensor:
    """
    Randomly choose an attack function.

    Parameters
    ----------
    images : tf.Tensor
        Images to randomly attack.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    attack = choices(list(attack_probs.keys()), weights=list(attack_probs.values()), k=1)[0]
    return attacks[attack](images)
