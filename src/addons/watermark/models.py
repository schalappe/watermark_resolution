# -*- coding: utf-8
"""
Network and subnetwork for watermark.
"""
from typing import Tuple

import keras
from keras import Model, layers

from src.addons.watermark.submodels import (
    embedding_stack,
    extract_stack,
    prepare_mark_stack,
    to_luma_chroma_stack,
    to_rgb_stack,
)


def create_watermark(image_dims: Tuple[int, int, int], mark_dims: Tuple[int, int, int], strength: float = 1.0) -> Model:
    """
    Create a Watermark network.

    Parameters
    ----------
    image_dims: Tuple[int, int, int]
        Dimension of image.
    mark_dims: Tuple[int, int, int]
        Dimension of mark.
    strength: float
        Strength scaling factor for controlling the watermark’s invisibility
        and the robustness against attacks.

    Returns
    -------
    Model
        Model for watermarking an image.
    """
    # ##: Image pre-processing network.
    inputs_image = keras.Input(shape=image_dims)
    luma, chroma = to_luma_chroma_stack(inputs_image)

    # ##: Mark pre-processing network.
    input_mark = keras.Input(shape=mark_dims)
    output_mark = prepare_mark_stack(input_mark, strength=strength)

    # ##: Embedding network.
    luma = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(luma)
    embedding = layers.concatenate([luma, output_mark], axis=-1)
    embedding = embedding_stack(embedding)

    # ##: Get watermark image.
    outputs = to_rgb_stack((embedding, chroma))

    return Model(inputs=[inputs_image, input_mark], outputs=outputs)


def create_extract_mark(image_dims: Tuple[int, int, int]) -> Model:
    """
    Create an ExtractWatermark network.

    Parameters
    ----------
     image_dims: Tuple[int, int, int]
        Dimension of image

    Returns
    -------
    Model
        Model for extracting watermark.
    """
    # ##: mark pre-processing network.
    inputs_image = keras.Input(shape=image_dims)
    luma, _ = to_luma_chroma_stack(inputs_image)

    # ##: extraction network.
    outputs = extract_stack(luma)

    return Model(inputs=inputs_image, outputs=outputs)
