# -*- coding: utf-8 -*-
"""
Set of functions for data augmentation.
"""
from imgaug.augmenters import RandAugment
import tensorflow as tf
from src.addons.augmenters.attacks import attacks


def augment(images: tf.Tensor) -> tf.Tensor:
    """
    Take an image and apply random data augmentation.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many image.

    Returns
    -------
    tf.Tensor
        Augmented images.
    """
    return RandAugment(n=2, m=15)(images=tf.cast(images, tf.uint8).numpy())


def apply_attacks(images: tf.Tensor, attack: str) -> tf.Tensor:
    """
    Apply a specific modification on image to deteriorate it.

    Parameters
    ----------
    images : tf.Tensor
         A tensor represented many image.
    attack : str
        Modification to apply.

    Returns
    -------
    tf.Tensor
        Deteriorated images.
    """
    if attack == "":
        return images
    return attacks[attack](images)
