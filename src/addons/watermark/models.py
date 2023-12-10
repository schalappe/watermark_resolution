# -*- coding: utf-8
"""
Network and subnetwork for watermark.
"""
import tensorflow as tf
import keras
from typing import Any, List, Tuple, Dict

from src.addons.images.color import RGBtoYCbCrLayer, YCbCrtoRGBLayer, SeparateYComponentLayer, CombineYCbCrLayer
from src.addons.watermark.metric import PeakSignalNoiseRatio, BitErrorRatio
from src.addons.watermark.layers import (
    ConvolutionLayer,
    TanhConvolutionLayer,
    UpScalingLayer,
    LastUpScalingLayer,
    XORScrambleLayer,
    Descaling,
)
from src.addons.augmenters.augment import random_attacks


def stack_mark_preprocessing(inputs: keras.layers.Layer, strength: float) -> keras.layers.Layer:
    """
    Add a stack of neural network for preprocessing.

    Parameters
    ----------
    inputs: keras.layers.Layer
        Layer to stack preprocessing.
    strength: float
        Strength factor to apply to the last layer.

    Returns
    -------
    keras.layers.Layer
        Layer after preprocessing.
    """
    block = XORScrambleLayer(key="110110")(inputs)
    block = UpScalingLayer(filters=512, kernel=3, strides=2)(block)
    for filters in [256, 128]:
        block = UpScalingLayer(filters=filters, kernel=3, strides=2)(block)
    block = LastUpScalingLayer(filters=1, kernel=3, strides=2)(block)
    block = tf.math.scalar_mul(strength, block)
    return block


def ycbcr_normalize(inputs: keras.layers.Layer) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
    """
    Transform RGB images to YCbCr image than normalize y-component.

    Parameters
    ----------
    inputs : keras.layers.Layer
        Layer to normalize.

    Returns
    -------
    Tuple[keras.layers.Layer, keras.layers.Layer]
        Y-component normalized and CbCr component.
    """
    image = keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    image = RGBtoYCbCrLayer()(image)
    y_component, chroma_component = SeparateYComponentLayer()(image)
    y_component = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(y_component)

    return y_component, chroma_component


def stack_image_preprocessing(inputs: keras.layers.Layer) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
    """
    Add a stack of neural network for preprocessing.

    Parameters
    ----------
    inputs: keras.layers.Layer
        Layer to stack preprocessing.

    Returns
    -------
    keras.layers.Layer
        Layer after preprocessing.
    """
    y_component, chroma_component = ycbcr_normalize(inputs)
    y_component = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(y_component)

    return y_component, chroma_component


def stack_embedding(inputs: keras.layers.Layer) -> keras.layers.Layer:
    """
    Add a stack of embedding layers to the previous layer.

    Parameters
    ----------
    inputs: keras.layers.Layer
        Layer to stack embedding.

    Returns
    -------
    keras.layers.Layer
        Layer after stacking embedding.
    """

    block = ConvolutionLayer(filters=64, kernel=3, strides=1)(inputs)
    for _ in range(3):
        block = ConvolutionLayer(filters=64, kernel=3, strides=1)(block)
    block = TanhConvolutionLayer(filters=1, kernel=3, strides=1)(block)

    return block


def stack_post_processing(inputs: Tuple[keras.layers.Layer, keras.layers.Layer]) -> keras.layers.Layer:
    """
    Add a stack of post-processing layers to the previous layer.

    Parameters
    ----------
    inputs: Tuple[keras.layers.Layer, keras.layers.Layer]
        Layer to stack post-processing.

    Returns
    -------
    keras.layers.Layer
        Layer after post-processing.
    """
    y_component, chroma_component = inputs
    y_component = Descaling(scale=1.0 / 127.5, offset=-1.0)(y_component)

    output = CombineYCbCrLayer()((y_component, chroma_component))
    output = YCbCrtoRGBLayer()(output)
    output = tf.cast(Descaling(scale=1.0 / 255, offset=0.0)(output), tf.uint8)

    return output


def create_watermark(image_dims: Tuple[int, int, int], mark_dims: Tuple[int, int, int], strength: float) -> keras.Model:
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
    keras.Model
        Model for watermarking an image.
    """
    # ##: Mark pre-processing network.
    input_mark = keras.Input(shape=mark_dims)
    output_mark = stack_mark_preprocessing(input_mark, strength=strength)

    # ##: Image pre-processing network.
    input_image = keras.Input(shape=image_dims)
    y_component, cbcr_component = stack_image_preprocessing(input_image)

    # ##: Embedding network.
    embedding = keras.layers.concatenate([y_component, output_mark], axis=-1)
    embedding = stack_embedding(embedding)

    # ##: Get watermark image.
    outputs = stack_post_processing((embedding, cbcr_component))

    # ##: return model.
    return keras.Model(inputs=[input_image, input_mark], outputs=outputs)


def create_extract_mark(image_dims: Tuple[int, int, int]) -> keras.Model:
    """
    Create an ExtractWatermark network.

    Parameters
    ----------
     image_dims: Tuple[int, int, int]
        Dimension of image

    Returns
    -------
    keras.Model
        Model for extracting watermark.
    """
    # ##: mark pre-processing network.
    input_image = keras.Input(shape=image_dims)
    y_component, _ = ycbcr_normalize(input_image)

    # ##: extraction network.
    for filters in [128, 256, 512]:
        y_component = ConvolutionLayer(filters=filters, kernel=3, strides=2)(y_component)
    outputs = TanhConvolutionLayer(filters=1, kernel=3, strides=2)(y_component)

    # ##: De-normalization and de-scrambling.
    outputs = Descaling(scale=1.0 / 127.5, offset=-1.0)(outputs)
    outputs = tf.cast(outputs, tf.uint8)
    outputs = XORScrambleLayer(key="110110")(outputs)

    # ##: return model.
    return keras.Model(inputs=input_image, outputs=outputs)


class WatermarkModel(keras.Model):
    """
    Model for embedding mark in images
    """

    def __init__(self, image_dims: Tuple[int, int, int], mark_dims: Tuple[int, int, int], strength: float = 1):
        super().__init__()
        self.watermark = create_watermark(image_dims=image_dims, mark_dims=mark_dims, strength=strength)
        self.extract_mark = create_extract_mark(image_dims=image_dims)
        self.loss_embedding_tracker = keras.metrics.Mean(name="loss_embedding")
        self.loss_extract_tracker = keras.metrics.Mean(name="loss_extract")
        self.psnr_tracker = PeakSignalNoiseRatio()
        self.ber_tracker = BitErrorRatio()

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """
        Return metrics.

        Returns
        -------
        List[keras.metrics.Metric]
            List of metrics.
        """
        return [self.loss_embedding_tracker, self.loss_extract_tracker, self.psnr_tracker, self.ber_tracker]

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, Any]:
        """
        Train method.

        Parameters
        ----------
        data
            One batch of data to be given to the loss function.

        Returns
        -------
        Dict[str, Any]
            Returns a dictionary with the loss metric.
        """
        images, marks = data

        with tf.GradientTape() as embedding_tape, tf.GradientTape() as extraction_tape:
            # ##: Embedding and marks.
            outputs = self.watermark([images, marks], training=True)
            attack_outputs = tf.cast(random_attacks()(outputs), tf.uint8)
            extracted_marks = self.extract_mark(attack_outputs, training=True)

            # ##: Calculates loss.
            loss_embedding, loss_extraction = self.loss([images, marks], [outputs, extracted_marks])

        # ##: computes gradient.
        grads_embedding = embedding_tape.gradient(loss_embedding, self.watermark.trainable_weights)
        grads_extraction = extraction_tape.gradient(loss_extraction, self.extract_mark.trainable_weights)

        # ##: train extraction model.
        self.optimizer.apply_gradients(zip(grads_embedding, self.watermark.trainable_weights))
        self.optimizer.apply_gradients(zip(grads_extraction, self.extract_mark.trainable_weights))

        # ##. update metrics
        self.loss_embedding_tracker.update_state(loss_embedding)
        self.loss_extract_tracker.update_state(loss_extraction)
        self.psnr_tracker.update_state(images, outputs)
        self.ber_tracker.update_state(marks, extracted_marks)
        return {
            "loss_embedding": self.loss_embedding_tracker.result(),
            "loss_extract": self.loss_extract_tracker.result(),
            "PSNR": self.psnr_tracker.result(),
            "BER": self.ber_tracker.result(),
        }
