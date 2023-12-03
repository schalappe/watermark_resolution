# -*- coding: utf-8
"""
    Network and subnetwork for watermark
"""
import tensorflow as tf
import keras
from typing import Any, List, Tuple


def block_pre_processing(block, filters: int, kernel: int, strides: int, last: bool = False) -> Any:
    """
    Sub-neural: Conv2DTranspose => BN => RELU => Pool[Average]

    if last: Conv2DTranspose => Pool[Average]

    Parameters
    ----------
    block: Any
        Input tensor.
    filters: int
        Filters of the conv layer.
    kernel: int
        Kernel size of the bottleneck layer.
    strides: int
        Stride for conv layer.
    last: bool
        if true, not normalize layer and activation.

    Returns
    -------
    Any
        Output tensor for the block.
    """
    block = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides)(block)
    if not last:
        block = keras.layers.BatchNormalization()(block)
        block = keras.layers.ReLU()(block)
    block = keras.layers.AveragePooling2D(pool_size=2, strides=1)(block)

    return block


def block_extract(block: Any, filters: int, kernel: int, strides: int) -> Any:
    """
    Sub-neural: Conv2D => BN => RELU

    Parameters
    ----------
    block: Any
        Input tensor.
    filters: int
        Filters of the conv layer.
    kernel: int
        Kernel size of the bottleneck layer.
    strides: int
        Stride for conv layer.

    Returns
    -------
    Any
        Output tensor for the block.
    """
    block = keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(block)
    block = keras.layers.BatchNormalization()(block)
    block = keras.layers.ReLU()(block)

    return block


def block_embedding(block: Any, filters: int, kernel: int, strides: int, last: bool = False) -> Any:
    """
    Sub-neural: Conv2D => BN => RELU

    if last: Conv2D => TANH

    Parameters
    ----------
    block: Any
        Input tensor.
    filters: int
        Filters of the conv layer
    kernel: int
        Kernel size of the bottleneck layer
    strides: int
        Stride for conv layer
    last: bool
        if true, not normalize layer and activation

    Returns
    -------
    Any
        Output tensor for the block.
    """
    block = keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(block)
    if not last:
        block = keras.layers.BatchNormalization()(block)
        block = keras.layers.ReLU()(block)
    else:
        block = keras.activations.tanh(block)

    return block


def stack_pre_processing(block: Any, blocks: List[int], kernel: int, strides: int) -> Any:
    """
    Stack of neural network.

    Parameters
    ----------
    block: Any
        input tensor
    blocks: List[int]
        Filters for conv layers
    kernel: int
        Kernel size of the bottleneck layer
    strides: int
        Stride for conv layer

    Returns
    -------
    Any
        Output tensor for the block.
    """
    for filters in blocks:
        block = block_pre_processing(block, filters=filters, kernel=kernel, strides=strides)
    return block


def stack_extract(block: Any, blocks: List[int], kernel: int, strides: int) -> Any:
    """
    Stack of neural network.

    Parameters
    ----------
    block: Any
        Input tensor
    blocks: List[int]
        Filters for conv layers
    kernel: int
        kernel size of the bottleneck layer
    strides: int
        Stride for conv layer

    Returns
    -------
    Any
        Output tensor for the block.
    """
    for filters in blocks:
        block = block_extract(block, filters=filters, kernel=kernel, strides=strides)
    return block


def create_watermark(image_dims: Tuple[int, int], mark_dims: Tuple[int, int], strength: float = 1) -> keras.Model:
    """
    Create a Watermark network.

    Parameters
    ----------
    image_dims: Tuple[int, int]
        Dimension of image
    mark_dims: Tuple[int, int]
        Dimension of mark
    strength: float
        Strength scaling factor for controlling the watermark’s invisibility
        and the robustness against attacks

    Returns
    -------
    keras.Model
        Model for watermarking an image
    """
    # ##: mark pre-processing network.
    input_mark = keras.Input(shape=mark_dims)
    output_mark = stack_pre_processing(input_mark, blocks=[512, 256, 128], kernel=3, strides=2)
    output_mark = block_pre_processing(output_mark, filters=1, kernel=3, strides=2, last=True)
    output_mark = tf.math.scalar_mul(strength, output_mark)

    # ##: image pre-processing network.
    input_image = keras.Input(shape=image_dims)
    output_imge = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(input_image)

    # ##: concatenate.
    embedding = keras.layers.concatenate([output_imge, output_mark], axis=-1)

    # ##: embedding network.
    block = block_embedding(embedding, filters=64, kernel=3, strides=1)
    for _ in range(3):
        block = block_embedding(block, filters=64, kernel=3, strides=1)
    outputs = block_embedding(block, filters=image_dims[-1], kernel=3, strides=1, last=True)

    # ##: return model.
    return keras.Model(inputs=[input_image, input_mark], outputs=outputs)


def create_extract_mark(mark_dims: Tuple[int, int]) -> keras.Model:
    """
    Create an ExtractWatermark network.

    Parameters
    ----------
    mark_dims: Tuple[int, int]
        Dimension of mark.

    Returns
    -------
    keras.Model
        Model for extracting watermark
    """
    # ##: mark pre-processing network.
    input_dims = (mark_dims[0] * 16, mark_dims[1] * 16, mark_dims[2])
    input_mark = keras.Input(shape=input_dims)

    # ##: extraction network.
    block = stack_extract(input_mark, blocks=[128, 256, 512], kernel=3, strides=2)
    outputs = keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation="tanh", padding="same")(block)

    # ##: return model.
    return keras.Model(inputs=input_mark, outputs=outputs)


class WaterMark:
    """
    Model for embedding mark in images
    """

    pass


class ExtractWaterMark:
    """
    Model for extracting mark in images
    """

    pass
