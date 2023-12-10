# -*- coding: utf-8 -*-
"""
Set of functions for input pipeline.
"""
from typing import Sequence

import tensorflow as tf

from src.addons.augmenters.augment import augment
from src.addons.images.load import load_image


def train_pipeline(paths: Sequence[str], batch: int) -> tf.data.Dataset:
    """
    Create data pipeline for training.

    Parameters
    ----------
    paths : Sequence[str]
        List of image path.
    batch : int
        Batch size.

    Returns
    -------
    tf.data.Dataset
        Pipeline for training.
    """
    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .shuffle(1024, seed=1335)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size=batch)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda image: image / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


def test_pipeline(paths: Sequence[str], batch: int) -> tf.data.Dataset:
    """
    Create data pipeline for evaluation.

    Parameters
    ----------
    paths : Sequence[str]
        List of image path.
    batch : int
        Batch size.
    attack : str, default=None
        Type of attack to apply on image.

    Returns
    -------
    tf.data.Dataset
        Pipeline for evaluation.
    """
    dataset = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size=batch)
        .cache()
    )

    return dataset.map(lambda image: image / 255.0, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )
