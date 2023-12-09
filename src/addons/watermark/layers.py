# -*- coding: utf-8 -*-
"""
Set of layers for watermark model.
"""
import tensorflow as tf
import keras


class XORScrambleLayer(keras.layers.Layer):
    """
    XOR operation as scrambling algorithm.
    """

    def __init__(self, key: str, **kwargs) -> None:
        super(XORScrambleLayer, self).__init__(**kwargs)

        self.key = key

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        XOR operation as scrambling algorithm.

        Parameters
        ----------
        inputs : tf.Tensor
            Data to scramble.

        Returns
        -------
        tf.Tensor
            Scrambled data.
        """
        flatten_data = tf.cast(tf.reshape(inputs, (-1, 1)), tf.int32)

        key_tensor = tf.constant(int(self.key, 2), dtype=tf.int32)
        scrambled_data = tf.bitwise.bitwise_xor(flatten_data, key_tensor)
        scrambled_data = tf.cast(tf.reshape(scrambled_data, tf.shape(inputs)), inputs.dtype)

        return scrambled_data


class NormalizationLayer(keras.layers.Layer):
    """
    De-normalization layer.
    """

    def __init__(self, scale, offset, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        De-normalization operation.

        Parameters
        ----------
        inputs : tf.Tensor
            Data to de-normalize.

        Returns
        -------
        tf.Tensor
            De-normalized data.
        """
        return (inputs - self.offset) / self.scale


class UpScalingLayer(keras.layers.Layer):
    """
    Stack of many neural layers.

    Conv2DTranspose => Batch Normalization => RELU => Average Pooling
    """

    def __init__(self, filters: int, kernel: int, strides: int, **kwargs):
        super(UpScalingLayer, self).__init__(**kwargs)

        self.conv = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides)
        self.norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.pool = keras.layers.AveragePooling2D(pool_size=2, strides=1)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural layers.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, default=False
            Whether is training or not.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        block = self.conv(inputs)
        block = self.norm(block, training=training)
        block = self.relu(block)
        block = self.pool(block)

        return block


class LastUpScalingLayer(keras.layers.Layer):
    """
    Stack of many neural layers.

    Conv2DTranspose => Average Pooling
    """

    def __init__(self, filters: int, kernel: int, strides: int, **kwargs):
        super(LastUpScalingLayer, self).__init__(**kwargs)

        self.conv = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides)
        self.pool = keras.layers.AveragePooling2D(pool_size=2, strides=1)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural layers.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        block = self.conv(inputs)
        block = self.pool(block)

        return block


class ConvolutionLayer(keras.layers.Layer):
    """
    Stack of multiple neural layers.

    Conv2D => Batch Normalization => ReLU
    """

    def __init__(self, filters: int, kernel: int, strides: int, **kwargs):
        super(ConvolutionLayer, self).__init__(**kwargs)

        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")
        self.norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural layers.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, default=False
            Whether is training or not.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        block = self.conv(inputs)
        block = self.norm(block, training=training)
        block = self.relu(block)

        return block


class TanhConvolutionLayer(keras.layers.Layer):
    """
    Stack of multiple neural layers.

    Conv2D => Tanh
    """

    def __init__(self, filters: int, kernel: int, strides: int, **kwargs):
        super(TanhConvolutionLayer, self).__init__(**kwargs)

        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")
        self.tanh = keras.activations.tanh

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Stack of multiple neural layers.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        block = self.conv(inputs)
        block = self.tanh(block)

        return block
