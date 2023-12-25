# -*- coding: utf-8
"""
Network for watermark.
"""
from datetime import datetime
from os.path import join
from typing import Any, Dict, Tuple

import keras
import tensorflow as tf
from keras import Model, layers
from keras.metrics import Mean
from keras.optimizers import Optimizer

from src.addons.data.augment import attacks, random_attacks
from src.addons.watermark.losses import WatermarkLoss
from src.addons.watermark.metrics import BitErrorRatio, PeakSignalNoiseRatio
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


class WatermarkModel(Model):
    """
    Watermark model.
    """

    def __init__(self, embedding: Model, extractor: Model, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = None
        self.embedding = embedding
        self.extractor = extractor

        self.signal_noise_metric = PeakSignalNoiseRatio()
        self.bit_error_metric = BitErrorRatio()
        self.loss_embedding_tracker = Mean("loss_embedding")
        self.loss_extractor_tracker = Mean("loss_extractor")

    @property
    def metrics(self):
        return [
            self.loss_embedding_tracker,
            self.loss_extractor_tracker,
            self.signal_noise_metric,
            self.bit_error_metric,
        ]

    @classmethod
    def from_ashes(cls, image_dims: Tuple[int, int, int], mark_dims: Tuple[int, int, int]):
        return WatermarkModel(
            embedding=create_watermark(image_dims, mark_dims), extractor=create_extract_mark(image_dims)
        )

    def compile(self, embedding_optimizer: Optimizer, extractor_optimizer: Optimizer, loss: Dict[str, Any]):
        """
        Configures the models for training.

        Parameters
        ----------
        embedding_optimizer : Optimizer
            Optimizer for training embeddings model.
        extractor_optimizer : Optimizer
            Optimizer for training extractor model.
        """
        super().compile()
        self.embedding.compile(optimizer=embedding_optimizer)
        self.extractor.compile(optimizer=extractor_optimizer)
        self.loss_fn = WatermarkLoss(**loss)

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        Single training step.

        Parameters
        ----------
        data : Tuple[tf.Tensor, tf.Tensor]
            Training data for training.
        """
        # ##: Split data and models.
        images, marks = data

        with tf.GradientTape(persistent=True) as tape:
            # ##: Embedding and marks.
            outputs_image = self.embedding([images, marks], training=True)
            outputs_mark = self.extractor(random_attacks(outputs_image), training=True)

            # ##: Calculates loss.
            loss_embedding, loss_extraction = self.loss_fn((images, marks), (outputs_image, outputs_mark))

        # ##: computes gradient.
        grads_embedding = tape.gradient(loss_embedding, self.embedding.trainable_variables)
        grads_extraction = tape.gradient(loss_extraction, self.extractor.trainable_variables)

        # ##: train extraction model.
        self.embedding.optimizer.apply_gradients(zip(grads_embedding, self.embedding.trainable_variables))
        self.extractor.optimizer.apply_gradients(zip(grads_extraction, self.extractor.trainable_variables))

        # ##: Metrics.
        self.signal_noise_metric.update_state(images, outputs_image)
        self.bit_error_metric.update_state(marks, outputs_mark)
        self.loss_embedding_tracker.update_state(loss_embedding)
        self.loss_extractor_tracker.update_state(loss_extraction)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def evaluate(self, test_set: tf.data.Dataset, attack: str = "identity") -> Tuple[float, float]:
        """
        Evaluate the model.

        Parameters
        ----------
        test_set : tf.data.Dataset
            Testing dataset.
        attack : str
            Attack to apply to watermark.

        Returns
        -------
        Tuple[float, float]
            Peak signal noise ratio and Bit error ratio.
        """
        self.signal_noise_metric.reset_states()
        self.bit_error_metric.reset_states()

        for images, marks in test_set:
            # ##: Predicted.
            outputs_image = self.embedding([images, marks], training=False)
            outputs_mark = self.extractor(attacks[attack](outputs_image), training=False)

            # ##: Metrics.
            self.signal_noise_metric.update_state(images, outputs_image)
            self.bit_error_metric.update_state(marks, outputs_mark)

        return self.signal_noise_metric.result(), self.bit_error_metric.result()

    def save(self, filepath: str):
        """
        Save the model.

        Parameters
        ----------
        filepath : str
            Path to save the models.
        """
        now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.embedding.save_weights(join(filepath, f"embedding.{now}.weights.h5"))
        self.extractor.save_weights(join(filepath, f"extractor.{now}.weights.h5"))
