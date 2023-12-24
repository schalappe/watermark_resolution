# -*- coding: utf-8 -*-
"""
Trains the neural network.
"""
from keras.optimizers import Optimizer
import tensorflow as tf
from typing import Tuple, Dict, Any

from src.addons.watermark.losses import WatermarkLoss
from src.addons.watermark.metrics import PeakSignalNoiseRatio, BitErrorRatio
from src.addons.watermark.models import create_watermark, create_extract_mark
from src.addons.data.augment import random_attacks, attacks


class WatermarkTrainer:
    """
    Class for training the watermark models.
    """

    def __init__(self):
        self.embedding = create_watermark(image_dims=(128, 128, 3), mark_dims=(8, 8, 1), strength=1.0)
        self.extractor = create_extract_mark(image_dims=(128, 128, 3))
        self.psnr_tracker = PeakSignalNoiseRatio()
        self.ber_tracker = BitErrorRatio()

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
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor], loss: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """
        Single training step.

        Parameters
        ----------
        data : Tuple[tf.Tensor, tf.Tensor]
            Training data for training.
        loss : Dict[str, Any]
            Loss configuration.

        Returns
        -------
        Dict[str, tf.Tensor]
            Results of training step.
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

        return {
            "loss_embedding": loss_embedding,
            "loss_extract": loss_extraction,
        }

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
        self.psnr_tracker.reset_states()
        self.ber_tracker.reset_states()

        # ##: Loop of train
        for batch, (images, marks) in enumerate(train_set):
            train_results = self.train_step(data=(images, marks), loss=loss)

            # ##: Print train result.
            loss_log = " — ".join([f"{key}: {float(value):.2f}" for key, value in train_results.items()])
            metric_log = " — ".join(
                [f"{metric.name}: {float(metric.result()):.2f}" for metric in [self.psnr_tracker, self.ber_tracker]]
            )
            print(
                "\033[K",
                f"Batch n°{batch+1} -> Losses: {loss_log} & Metrics: {metric_log}",
                sep="",
                end="\r",
                flush=True,
            )
        print("\n")

        return self.psnr_tracker.result(), self.ber_tracker.result()

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
