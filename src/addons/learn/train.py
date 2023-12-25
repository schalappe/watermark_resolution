# -*- coding: utf-8 -*-
"""
Trains the neural network.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import keras
import tensorflow as tf
from keras import Model
from keras.optimizers import Optimizer
from keras.metrics import Mean

from src.addons.data.augment import attacks, random_attacks
from src.addons.watermark.losses import WatermarkLoss
from src.addons.watermark.metrics import BitErrorRatio, PeakSignalNoiseRatio
from src.addons.watermark.models import create_extract_mark, create_watermark


class Watermark:
    """
    Class for training the watermark models.
    """

    def __init__(self, embedding: Model, extractor: Model):
        self.embedding = embedding
        self.extractor = extractor
        self.psnr_tracker = PeakSignalNoiseRatio()
        self.ber_tracker = BitErrorRatio()
        self.loss_embedding_tracker = Mean("loss_embedding")
        self.loss_extractor_tracker = Mean("loss_extractor")

    @classmethod
    def from_storage(cls, embedding_path, extractor_path) -> Watermark:
        return Watermark(
            embedding=keras.saving.load_model(embedding_path), extractor=keras.saving.load_model(extractor_path)
        )

    @classmethod
    def create(cls) -> Watermark:
        return Watermark(
            embedding=create_watermark(image_dims=(128, 128, 3), mark_dims=(8, 8, 1), strength=1.0),
            extractor=create_extract_mark(image_dims=(128, 128, 3)),
        )

    def compile(self, embedding_optimizer: Optimizer, extractor_optimizer: Optimizer):
        """
        Configures the models for training.

        Parameters
        ----------
        embedding_optimizer : Optimizer
            Optimizer for training embeddings model.
        extractor_optimizer : Optimizer
            Optimizer for training extractor model.
        """
        self.embedding.compile(optimizer=embedding_optimizer)
        self.extractor.compile(optimizer=extractor_optimizer)

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor], loss: Dict[str, Any]):
        """
        Single training step.

        Parameters
        ----------
        data : Tuple[tf.Tensor, tf.Tensor]
            Training data for training.
        loss : Dict[str, Any]
            Loss configuration.
        """
        # ##: Split data and models.
        images, marks = data

        with tf.GradientTape(persistent=True) as tape:
            # ##: Embedding and marks.
            outputs_image = self.embedding([images, marks], training=True)
            outputs_mark = self.extractor(random_attacks(outputs_image), training=True)

            # ##: Calculates loss.
            loss_embedding, loss_extraction = WatermarkLoss(**loss)((images, marks), (outputs_image, outputs_mark))

        # ##: computes gradient.
        grads_embedding = tape.gradient(loss_embedding, self.embedding.trainable_variables)
        grads_extraction = tape.gradient(loss_extraction, self.extractor.trainable_variables)

        # ##: train extraction model.
        self.embedding.optimizer.apply_gradients(zip(grads_embedding, self.embedding.trainable_variables))
        self.extractor.optimizer.apply_gradients(zip(grads_extraction, self.extractor.trainable_variables))

        # ##: Metrics.
        self.psnr_tracker.update_state(images, outputs_image)
        self.ber_tracker.update_state(marks, outputs_mark)
        self.loss_embedding_tracker.update_state(loss_embedding)
        self.loss_extractor_tracker.update_state(loss_extraction)

    def learn(self, train_set: tf.data.Dataset, loss: Dict[str, float]) -> Tuple[float, float]:
        """
        Train a Watermark neural network on a dataset.

        Parameters
        ----------
        train_set : tf.data.Dataset
            Training dataset.
        loss : Dict[str, float]
            Loss configuration.

        Returns
        -------
        Tuple[float, float]
            Peak signal noise ratio and Bit error ratio.
        """
        # ##: Metrics.
        self.psnr_tracker.reset_states()
        self.ber_tracker.reset_states()
        self.embedding.reset_states()
        self.extractor.reset_states()

        # ##: Loop of train
        for batch, (images, marks) in enumerate(train_set):
            self.train_step(data=(images, marks), loss=loss)

            # ##: Print train result.
            metric_log = " — ".join(
                [
                    f"{metric.name}: {float(metric.result()):.2f}"
                    for metric in [
                        self.loss_embedding_tracker,
                        self.loss_extractor_tracker,
                        self.psnr_tracker,
                        self.ber_tracker,
                    ]
                ]
            )
            print("\033[K", f"Batch n°{batch+1} ->{metric_log}", sep="", end="\r", flush=True)
        print("B")

        return self.loss_embedding_tracker.result(), self.loss_extractor_tracker.result()

    def fit(self, train_set: tf.data.Dataset, epochs: int, loss: Dict[str, Any], early_stopping: bool = True):
        """
        Train the models.

        Parameters
        ----------
         train_set : tf.data.Dataset
            Training dataset.
        epochs : int
            Epoch number.
        loss : Dict[str, float]
            Loss configuration.
        """
        # ##: Early stop parameters.
        patience = 5
        wait = 0
        best_emb, best_extr = float("-inf"), float("inf")

        for epoch in range(epochs):
            print(f"Epoch n°{epoch+1}")
            loss_emb, loss_extr = self.learn(train_set=train_set, loss=loss)

            # ##: The early stopping strategy.
            if early_stopping:
                wait += 1
                if loss_extr < best_extr and loss_emb > best_emb:
                    wait = 0

                best_emb = loss_emb if loss_emb > best_emb else best_emb
                best_extr = loss_extr if loss_extr < best_extr else best_extr

                if wait >= patience:
                    break

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
        self.psnr_tracker.reset_states()
        self.ber_tracker.reset_states()

        for images, marks in test_set:
            # ##: Predicted.
            outputs_image = self.embedding([images, marks], training=False)
            outputs_mark = self.extractor(attacks[attack](outputs_image), training=False)

            # ##: Metrics.
            self.psnr_tracker.update_state(images, outputs_image)
            self.ber_tracker.update_state(marks, outputs_mark)

        return self.psnr_tracker.result(), self.ber_tracker.result()
