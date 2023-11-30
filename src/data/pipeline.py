# -*- coding: utf-8 -*-
"""
Set of functions for input pipeline.
"""
from typing import Tuple, Sequence

import tensorflow as tf

from src.addons.augmenters.augment import augment
from src.addons.images.load import load_image


def train_pipeline(paths: Sequence[str], batch: int, dims: Tuple[int, int]) -> tf.data.Dataset:
    """
    Create data pipeline for training.

    Parameters
    ----------
    paths : Sequence[str]
        List of image path.
    batch : int
        Batch size.
    dims : Tuple[int, int]
        Height and width of image use for training.

    Returns
    -------
    tf.data.Dataset
        Pipeline for training.
    """
    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .shuffle(1024, seed=1335)
        .map(lambda path: load_image(path, height=dims[0], width=dims[0]), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size=batch)
        .map(lambda image: tf.py_function(augment, [image], [tf.float32])[0], num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


def test_pipeline(paths: Sequence[str], batch: int, dims: Tuple[int, int]) -> tf.data.Dataset:
    """
    Create data pipeline for evaluation.

    Parameters
    ----------
    paths : Sequence[str]
        List of image path.
    batch : int
        Batch size.
    dims : Tuple[int, int]
        Height and width of image use for evaluation.

    Returns
    -------
    tf.data.Dataset
        Pipeline for evaluation.
    """
    return (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda path: load_image(path, height=dims[0], width=dims[0]), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size=batch)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
