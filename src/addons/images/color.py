# -*- coding: utf-8 -*-
"""
Set of functions for changing color space.
"""
import tensorflow as tf
from typing import Tuple
from keras import layers


class RGBtoYCbCrLayer(layers.Layer):
    """
    Transform RGB to YCbCr.
    """

    def __init__(self, **kwargs):
        super(RGBtoYCbCrLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Transformation RGB image to YCbCr image.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented an image with a RGB color range.

        Returns
        -------
        tf.Tensor
            Tensor represented an image with a YCbCr color range.
        """
        # ##: Assuming inputs is an RGB image tensor in the range [0, 1].
        r = inputs[:, :, :, 0]
        g = inputs[:, :, :, 1]
        b = inputs[:, :, :, 2]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)

        # ##: Stack the Y, Cb, and Cr components
        y_cb_cr = tf.stack([y, cb, cr], axis=-1)

        return y_cb_cr


class YCbCrtoRGBLayer(layers.Layer):
    """
    Transform YCbCr to RGB.
    """

    def __init__(self, **kwargs):
        super(YCbCrtoRGBLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Transform YCbCr image to RGB image.

        Parameters
        ----------
        inputs :  tf.Tensor
            Tensor represented an image with a YCbCr color range.

        Returns
        -------
        tf.Tensor
            Tensor represented an image with a RGB color range.
        """
        # ##: Assuming inputs is a YCbCr image tensor
        y = inputs[:, :, :, 0]
        cb = inputs[:, :, :, 1]
        cr = inputs[:, :, :, 2]

        r = y + 1.403 * cr
        g = y - 0.714 * cr - 0.344 * cb
        b = y + 1.773 * cb

        # ##: Stack the R, G, and B components
        rgb = tf.stack([r, g, b], axis=-1)

        return rgb


class SeparateYComponentLayer(layers.Layer):
    """
    Extract Y component and CbCr component from YCbCr image.
    """

    def __init__(self, **kwargs):
        super(SeparateYComponentLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Separate YCbCr image into Y component and CbCr component.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented an image with a YCbCr color range.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Y component and CbCr component.
        """
        y = inputs[:, :, :, 0:1]
        cb_cr = inputs[:, :, :, 1:]

        return y, cb_cr


class CombineYCbCrLayer(layers.Layer):
    """
    Combine Y component and CbCr component into one YCbCr.
    """

    def __init__(self, **kwargs):
        super(CombineYCbCrLayer, self).__init__(**kwargs)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Combine Y component and CbCr component into one YCbCr image.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Y component and CbCr component.

        Returns
        -------
        tf.Tensor
            Tensor represented an image with a YCbCr color range.
        """
        y, cb_cr = inputs
        return tf.concat([y, cb_cr], axis=-1)
