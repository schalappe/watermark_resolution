# -*- coding: utf-8 -*-
"""
Set of function for input pipeline
"""
from glob import glob
from os.path import join
from typing import Tuple

import tensorflow as tf

from src.processors import load_image, preprocess_input


def prepare_data_from_slice(
    inputs_path: str, extension: str, batch_size: int, training: bool = True
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a Dataset for training

    Parameters
    ----------
    inputs_path: str
        Path of images for training
    extension: str
        Extension of images
    batch_size: int
        Batch size
    training: bool
        If training, shuffle data

    Returns
    -------
        Dataset and len of dataset
    """
    # check image extension
    if extension.lower() not in ["png", "jpeg", "jpg", "gif"]:
        raise NotImplementedError

    # load images path
    image_paths = glob(join(inputs_path, "*." + extension))

    # build the dataset and data input pipeline
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # shuffle data
    if training:
        dataset = dataset.shuffle(1024)

    # load image and preprocess
    dataset = dataset.map(
        lambda image: preprocess_input(inputs=load_image(image), mode="tf"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # cache
    dataset = dataset.cache()

    # Batch all datasets
    dataset = dataset.batch(batch_size)

    # Use buffered prefetching on all datasets
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE), len(image_paths)
