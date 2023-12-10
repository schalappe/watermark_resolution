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
    GaussianBlur,
    Rotate,
    CropAndPad,
    SaltAndPepper,
    JpegCompression,
)
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
    return Rotate(rotate=(0, 90))(images=tf.cast(images, tf.uint8).numpy())


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
    return Dropout(p=(0.3, 0.9))(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
def random_crop(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random crop.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Images with crop.
    """
    return CropAndPad(percent=(0.5, 0.8))(images=tf.cast(images, tf.uint8).numpy())


@tf.py_function(Tout=tf.float32)
def random_gaussian_blur(images: tf.Tensor) -> tf.Tensor:
    """
    Apply random gaussian blur.

    Parameters
    ----------
    images: tf.Tensor
         A tensor represented an image.

    Returns
    -------
    tf.Tensor
        Blurred images.
    """
    return GaussianBlur(sigma=(0.0, 3.0))(images=tf.cast(images, tf.uint8).numpy())


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
    return AverageBlur(k=(3, 5))(images=tf.cast(images, tf.uint8).numpy())


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
    return MedianBlur(k=(3, 5))(images=tf.cast(images, tf.uint8).numpy())


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
    return AdditiveGaussianNoise(scale=0.1 * 225)(images=tf.cast(images, tf.uint8).numpy())


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
    return SaltAndPepper(p=0.1)(images=tf.cast(images, tf.uint8).numpy())


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
    return JpegCompression(compression=50)(images=tf.cast(images, tf.uint8).numpy())
