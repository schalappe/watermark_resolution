# -*- coding: utf-8 -*-
"""
Set of functions to modify an image.
"""
from imgaug.augmenters import (
    Fliplr,
    Flipud,
    AddToHue,
    AddToSaturation,
    AddToBrightness,
    GammaContrast,
    AdditiveGaussianNoise,
    AverageBlur,
    Dropout,
    MedianBlur,
    Rotate,
    SaltAndPepper,
    JpegCompression,
)
from random import choice
import tensorflow as tf


@tf.py_function(Tout=tf.float32)
def random_flip(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random flip to images.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many images.

    Returns
    -------
    tf.Tensor
        Flipped images.
    """
    aug_image = Fliplr(0.5)(images=tf.cast(images, tf.uint8).numpy())
    aug_image = Flipud(0.5)(images=tf.cast(aug_image, tf.uint8).numpy())
    return aug_image


@tf.py_function(Tout=tf.float32)
def random_hue(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random hue to images.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many images.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    return AddToHue([-50, 50])(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
def random_saturation(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random hue to images.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many images.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    return AddToSaturation((-50, 50))(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
def random_brightness(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random hue to images.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many images.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    return AddToBrightness((-30, 30))(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
def random_contrast(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random contrast to images.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many images.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    return GammaContrast((0.5, 2.0))(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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


@tf.py_function(Tout=tf.float32)
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
    return JpegCompression(compression=value)(images=tf.cast(images, tf.uint8).numpy())
