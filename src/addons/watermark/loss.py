# -*- coding: utf-8
"""
Custom losses
"""
from typing import Tuple, Dict, Any
import tensorflow as tf
import keras


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

    def call(
        self, y_true: Tuple[tf.Tensor, tf.Tensor], y_pred: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculates losses for watermark network;

        Parameters
        ----------
        y_true : Tuple[tf.Tensor, tf.Tensor]
            Ground truth image and Ground truth mark.
        y_pred : Tuple[tf.Tensor, tf.Tensor]
            Image predicted and Mark predicted

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            `L_emb = λ_1 * L_1 + λ_2 * L_2`, `L_ext = λ_3 * L_2`
        """
        true_image, true_mark = y_true
        pred_image, pred_mark = y_pred

        # middle losses.
        loss_image = tf.reduce_mean(tf.math.squared_difference(true_image, pred_image), axis=[-3, -2, -1])
        loss_mark = tf.reduce_mean(tf.math.abs(true_mark - pred_mark), axis=[-3, -2, -1])

        # final losses.
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
