# -*- coding: utf-8 -*-
"""
Set of sub-models used in the watermark.
"""
from typing import Tuple
from keras import layers
from keras.layers import Layer, Rescaling, Lambda
from src.addons.watermark.layers import (
    YCbCrToRGB,
    RGBToYCbCr,
    SplitLumaChroma,
    InverseRescaling,
    UpSampling,
    ReluConvolution,
    TanhConvolution,
    ArnoldCat,
    InverseArnoldCat,
)


def to_luma_chroma_stack(inputs: Layer) -> Tuple[Layer, Layer]:
    """
    Transform RGB inputs to YCbCr outputs than normalize luma component.

    Parameters
    ----------
    inputs : Layer
        Layer to stack.

    Returns
    -------
    Tuple[Layer, Layer]
        Y-component normalized and CbCr component.
    """
    hidden = Rescaling(scale=1.0 / 255)(inputs)  # ##: Rescale [0,255] to [0, 1]
    hidden = RGBToYCbCr()(hidden)
    luma, chroma = SplitLumaChroma()(hidden)
    luma = Rescaling(scale=2.0, offset=-1.0)(luma)  # ##: Rescale [0,1] to [-1, 1]

    return luma, chroma


def to_rgb_stack(inputs: Tuple[Layer, Layer]) -> Layer:
    """
    Transform YCbCr inputs to RGB outputs than de-normalize it.

    Parameters
    ----------
    inputs : Tuple[Layer, Layer]
        Layer to stack.

    Returns
    -------
    Layer
        RGB de-normalized.
    """
    luma, chroma = inputs
    luma = InverseRescaling(scale=2.0, offset=-1.0)(luma)  # ##: Rescale [-1, 1] to [0,1]
    hidden = layers.concatenate([luma, chroma], axis=-1)
    hidden = YCbCrToRGB()(hidden)
    return InverseRescaling(scale=1.0 / 255, offset=0.0)(hidden)  # ##: Rescale [0, 1] to [0,255]


def prepare_mark_stack(inputs: Layer, strength: float = 1.0) -> Layer:
    """
    Stack multiple layers of neural networks for preprocessing mark images.

    Parameters
    ----------
    inputs : Layer
        Layer to stack.
    strength : float, default=1.0
        Strength factor to apply to the last layer.

    Returns
    -------
    Layer
        Final layer of stacked layers.
    """
    hidden = ArnoldCat(iterations=1)(inputs)
    hidden = UpSampling(filters=512, kernel=3, strides=2)(hidden)
    hidden = UpSampling(filters=256, kernel=3, strides=2)(hidden)
    hidden = UpSampling(filters=128, kernel=3, strides=2)(hidden)
    hidden = UpSampling(filters=1, kernel=3, strides=2)(hidden, last=True)
    return Lambda(lambda tensor: tensor * strength)(hidden)


def embedding_stack(inputs: Layer) -> Layer:
    """
    Stack multiple layers of neural networks for embedding images and marks.

    Parameters
    ----------
    inputs : Layer
        Layer to stack.

    Returns
    -------
    Layer
        Final layer of stacked layers.
    """
    hidden = ReluConvolution(filters=64, kernel=3, strides=1)(inputs)
    for _ in range(3):
        hidden = ReluConvolution(filters=64, kernel=3, strides=1)(hidden)

    return TanhConvolution(filters=1, kernel=3, strides=1)(hidden)


def extract_stack(inputs: Layer) -> Layer:
    """
    Stack multiple layers of neural networks for extracting marks.

    Parameters
    ----------
    inputs : Layer
        Layer to stack.

    Returns
    -------
    Layer
        Final layer of stacked layers.
    """
    hidden = ReluConvolution(filters=128, kernel=3, strides=2)(inputs)
    for filters in [256, 512]:
        hidden = ReluConvolution(filters=filters, kernel=3, strides=2)(hidden)
    hidden = TanhConvolution(filters=1, kernel=3, strides=2)(hidden)
    hidden = InverseRescaling(scale=2.0, offset=-1.0)(hidden)  # ##: Rescale [-1, 1] to [0,1]

    return InverseArnoldCat(iterations=1)(hidden)
