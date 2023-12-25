# -*- coding: utf-8 -*-
"""
Set of layers for watermark model.
"""
from typing import Tuple

import tensorflow as tf
from keras import layers
from keras.layers import Activation, AveragePooling2D, Layer


class RGBToYCbCr(Layer):
    """
    Convert RGB images to YCbCr images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = tf.constant(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Convert RGB images to YCbCr images.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented a RGB images

        Returns
        -------
        tf.Tensor
            Tensor represented a YCbCr images.
        """
        return tf.tensordot(inputs, tf.transpose(self.kernel), axes=((-1,), (0,)))


class YCbCrToRGB(Layer):
    """
    Convert YCbCr images to RGB images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = tf.constant([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]], dtype=tf.float32)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Convert YCbCr images to RGB images.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented a YCbCr images.

        Returns
        -------
        tf.Tensor
            Tensor represented a RGB images.
        """
        return tf.tensordot(inputs, tf.transpose(self.kernel), axes=((-1,), (0,)))


class InverseRescaling(Layer):
    """
    Inverse rescaling operation.
    """

    def __init__(self, scale, offset, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Inverse rescaling operation on the input data.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented data.

        Returns
        -------
        tf.Tensor
            Result of the inverse rescaling operation.
        """
        return (inputs - tf.cast(self.offset, dtype=inputs.dtype)) / tf.cast(self.scale, inputs.dtype)


class SplitLumaChroma(Layer):
    """
    Split YCbCr into Luma and Chroma components.
    """

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split YCbCr into Luma and Chroma components.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented data.

        Returns
        -------
        tf.Tensor, tf.Tensor
            Luma components and chroma components.
        """
        return inputs[:, :, :, 0:1], inputs[:, :, :, 1:]


class UpSampling(Layer):
    """
    Upsampling layer.
    """

    def __init__(self, filters, kernel, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2DTranspose(filters, kernel, strides=strides)
        self.norm = layers.BatchNormalization()
        self.pool = AveragePooling2D(pool_size=2, strides=1)

    def call(self, inputs: tf.Tensor, last=False, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural network layers:
            - Conv2DTranspose => Batch Normalization => RELU => Average Pooling
            - if last is True: Conv2DTranspose => Average Pooling.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.
        last : bool, default=False
            Last layer or not.

        Returns
        -------
        tf.Tensor
            Output of the upsampling layer.
        """
        hidden = self.conv(inputs)
        if last is False:
            hidden = self.norm(hidden)
            hidden = Activation("relu")(hidden)
        return self.pool(hidden)


class ReluConvolution(Layer):
    """
    Stack of convolution layers and relu activation.
    """

    def __init__(self, filters, kernel, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel, strides=strides, padding="same")
        self.norm = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural network layers: Conv2D => Batch Normalization => ReLU.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the convolution layer.
        """
        hidden = self.conv(inputs)
        hidden = self.norm(hidden)
        return Activation("relu")(hidden)


class TanhConvolution(Layer):
    """
    Stack of convolution layers and tanh activation.
    """

    def __init__(self, filters, kernel, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel, strides=strides, padding="same")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural network layers: Conv2D => Tanh.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the convolution layer.
        """
        hidden = self.conv(inputs)
        return Activation("tanh")(hidden)


class ArnoldCat(Layer):
    """
    Arnold cat algorithm for image scrambling.
    """

    def __init__(self, iterations: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.iterations = int(iterations)

    def scramble(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Scramble an input data with arnold cat algorithm.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the arnold cat algorithm.
        """
        height, width = tf.shape(inputs)[0], tf.shape(inputs)[1]

        for _ in range(self.iterations):
            x, y = tf.meshgrid(tf.range(width), tf.range(height))
            new_x = tf.math.floormod(2 * x + y, width)
            new_y = tf.math.floormod(x + y, height)
            indices = tf.transpose(tf.stack([new_y, new_x], axis=0))
            inputs = tf.gather_nd(inputs, indices)

        return inputs

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Scramble an input data with arnold cat algorithm.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the arnold cat algorithm.
        """
        return tf.map_fn(self.scramble, inputs)


class InverseArnoldCat(Layer):
    """
    Arnold cat algorithm for image scrambling.
    """

    def __init__(self, iterations: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.iterations = int(iterations)

    def descramble(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        De-scramble an input data with arnold cat algorithm.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the arnold cat algorithm.
        """
        height, width = tf.shape(inputs)[0], tf.shape(inputs)[1]

        for _ in range(self.iterations):
            x, y = tf.meshgrid(tf.range(width), tf.range(height))
            new_x = tf.math.floormod(-2 * x + y, width)
            new_y = tf.math.floormod(x - y, height)
            indices = tf.transpose(tf.stack([new_y, new_x], axis=0))
            inputs = tf.gather_nd(inputs, indices)

        return inputs

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        De-scramble an input data with arnold cat algorithm.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor represented input data.

        Returns
        -------
        tf.Tensor
            Output of the arnold cat algorithm.
        """
        return tf.map_fn(self.descramble, inputs)
