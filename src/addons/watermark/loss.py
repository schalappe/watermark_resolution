# -*- coding: utf-8
"""
Custom losses
"""
from typing import Tuple, Dict, Any
import tensorflow as tf
import keras


def mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate mean squared error between two tensors.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth image.
    y_pred : tf.Tensor
        Image predicted.

    Returns
    -------
    tf.Tensor
        Mean squared error between two tensors.
    """
    batch = tf.shape(y_true)[0]
    squared = tf.square(y_true - y_pred)
    sum_quadratic = tf.reduce_sum(tf.reshape(squared, shape=(batch, -1)), axis=-1)
    return tf.reduce_mean(sum_quadratic)


def mean_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate mean absolute error between two tensors.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth image.
    y_pred : tf.Tensor
        Image predicted.

    Returns
    -------
    tf.Tensor
        Mean absolute error between two tensors.
    """
    batch = tf.shape(y_true)[0]
    absolute = tf.abs(y_true - y_pred)
    sum_absolute = tf.reduce_sum(tf.reshape(absolute, shape=(batch, -1)))
    return tf.reduce_mean(sum_absolute)


def watermark_loss(
    y_true: Tuple[tf.Tensor, tf.Tensor],
    y_pred: Tuple[tf.Tensor, tf.Tensor],
    strength_embedding_mse: float = 45.0,
    strength_embedding_mae: float = 0.2,
    strength_extraction_mae: float = 20.0,
) -> Tuple[tf.Tensor, ...]:
    """
    Calculates the loss for watermark neural network.

    Parameters
    ----------
    y_true : Tuple[tf.Tensor, tf.Tensor]
        Ground truth image and ground truth mark.
    y_pred : Tuple[tf.Tensor, tf.Tensor, tf]
        Image predicted and mark predicted.
    strength_embedding_mse : float, optional (default=45.0)
    strength_embedding_mae : float, optional (default=0.2)
    strength_extraction_mae : float, optional (default=20)

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        `L_emb = λ_1 * L_1 + λ_2 * L_2`
        `L_ext = λ_3 * L_2`
    """
    # ##: Split arguments.
    true_image, true_mark = y_true
    pred_image, pred_mark = y_pred

    # ##: middle losses.
    loss_image = mean_squared_error(true_image, pred_image)
    loss_mark = mean_absolute_error(true_mark, pred_mark)

    # ##: final losses.
    loss_embedding = tf.math.scalar_mul(strength_embedding_mse, loss_image) + tf.math.scalar_mul(
        strength_embedding_mae, loss_mark
    )

    loss_extraction = tf.math.scalar_mul(strength_extraction_mae, loss_mark)

    return loss_embedding, loss_extraction


class WatermarkLoss(keras.losses.Loss):
    """
    Loss function for watermark.

    Parameters
    ----------
    strength_embedding_mse : float
        strength of the L1 loss applied to the embedding network
    strength_embedding_mae : float
        strength of the L2 loss applied to the embedding network
    strength_extraction_mae : float
         strength of the L2 loss applied to the extraction network
    """

    def __init__(
        self,
        strength_embedding_mse: float = 45.0,
        strength_embedding_mae: float = 0.2,
        strength_extraction_mae: float = 20,
    ):
        self.strength_embedding_mse = strength_embedding_mse
        self.strength_embedding_mae = strength_embedding_mae
        self.strength_extraction_mae = strength_extraction_mae
        super().__init__(name="watermark_loss")

    def __call__(
        self, true_image: tf.Tensor, true_mark: tf.Tensor, pred_image: tf.Tensor, pred_mark: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculates losses for watermark network.

        Parameters
        ----------
        true_image : tf.Tensor
            Ground truth image.
        true_mark : tf.Tensor
            Ground truth mark.
        pred_image : Tuple[tf.Tensor, tf.Tensor]
            Image predicted.
        pred_mark : tf.Tensor
            Mark predicted.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            `L_emb = λ_1 * L_1 + λ_2 * L_2`
            `L_ext = λ_3 * L_2`
        """
        # ##: middle losses.
        loss_image = tf.reduce_mean(tf.math.squared_difference(true_image, pred_image), axis=[-3, -2, -1])
        loss_mark = tf.reduce_mean(tf.math.abs(true_mark - pred_mark), axis=[-3, -2, -1])

        # ##: final losses.
        loss_embedding = self.strength_embedding_mse * loss_image + self.strength_embedding_mae * loss_mark
        loss_extraction = self.strength_extraction_mae * loss_mark

        return tf.reduce_mean(loss_embedding), tf.reduce_mean(loss_extraction)

    def get_config(self) -> Dict[str, Any]:
        """ "
        Returns the config dictionary for a `WatermarkLoss` instance.

        Returns
        -------
        Dict[str, Any]
            Configuration for `WatermarkLoss` instance.
        """
        base_config = super().get_config()
        return {
            **base_config,
            "strength_embedding_mse": self.strength_embedding_mse,
            "strength_embedding_mae": self.strength_embedding_mae,
            "strength_extraction_mae": self.strength_extraction_mae,
        }
