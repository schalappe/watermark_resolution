# -*- coding: utf-8 -*-
"""
Set of functions for modifying data.
"""
from random import choice

import tensorflow as tf
from imgaug.augmenters import (
    AdditiveGaussianNoise,
    AverageBlur,
    Dropout,
    MedianBlur,
    Rotate,
    SaltAndPepper,
)


def random_rotate(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random rotation.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Rotated images.
    """
    angle = choice([15 * i for i in range(1, 6)])
    return Rotate(rotate=angle)(images=tf.cast(images, tf.uint8).numpy())


def random_dropout(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random dropout.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Images with dropout.
    """
    value = choice([0.1, 0.3, 0.5])
    return Dropout(p=value)(images=tf.cast(images, tf.uint8).numpy())


def random_average_blur(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random average blur.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Blurred images.
    """
    kernel = choice([3, 5])
    return AverageBlur(k=kernel)(images=tf.cast(images, tf.uint8).numpy())


def random_median_blur(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random dropout.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Blurred images.
    """
    kernel = choice([3, 5])
    return MedianBlur(k=kernel)(images=tf.cast(images, tf.uint8).numpy())


def random_gaussian_noise(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random gaussian noise.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Noised images.
    """
    sigma = choice([0.01, 0.03])
    return AdditiveGaussianNoise(scale=sigma * 225)(images=tf.cast(images, tf.uint8).numpy())


def random_salt_pepper(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random gaussian noise.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Noised images.
    """
    value = choice([0.01, 0.03, 0.05])
    return SaltAndPepper(p=value)(images=tf.cast(images, tf.uint8).numpy())


def random_jpeg_quality(images: tf.Tensor) -> tf.Tensor:
    """
    Randomly reduce JPEG quality.

    Parameters
    ----------
    images: tf.Tensor
        A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Reduced images.
    """
    # value of rotation
    value = choice([50, 70, 90])

    if len(images.shape) == 3:
        return tf.image.random_jpeg_quality(images, value - 1, value)

    return tf.map_fn(
        fn=lambda image: tf.image.random_jpeg_quality(image, value - 1, value),
        elems=images,
    )


attacks = {
    "rotation": random_rotate,
    "dropout": random_dropout,
    "average_blur": random_average_blur,
    "median_blur": random_median_blur,
    "gaussian_noise": random_gaussian_noise,
    "salt_pepper": random_salt_pepper,
    "image_quality": random_jpeg_quality,
}
