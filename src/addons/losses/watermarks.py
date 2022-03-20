# -*- coding: utf-8
"""
Custom losses
"""
import tensorflow as tf


class WatermarkLoss(tf.keras.losses.Loss):
    """
    Create watermark loss function

    Parameters
    ----------
    strength_embedding_mse: float
        strength of the L1 loss applied to the embedding network

    strength_embedding_mae: float
        strength of the L2 loss applied to the embedding network

    strength_extraction_mae: float
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
        self,
        y_true_image: tf.Tensor,
        y_true_mark: tf.Tensor,
        y_pred_image: tf.Tensor,
        y_pred_mark: tf.Tensor,
    ):
        """
        Calculates losses for watermark network
        Parameters
        ----------
        y_true_image: tf.Tensor
            Ground truth image

        y_true_mark: tf.Tensor
            Ground truth mark

        y_pred_image: tf.Tensor
            Image predicted

        y_pred_mark: tf.Tensor
            Mark predicted

        Returns
        -------
            `L_emb = λ_1 * L_1 + λ_2 * L_2`, `L_ext = λ_3 * L_2`
        """
        # middle losses
        loss_image = tf.reduce_mean(
            tf.math.squared_difference(y_true_image, y_pred_image), axis=[-3, -2, -1]
        )
        loss_mark = tf.reduce_mean(
            tf.math.abs(y_true_mark - y_pred_mark), axis=[-3, -2, -1]
        )

        # final losses
        loss_embedding = (
            self.strength_embedding_mse * loss_image
            + self.strength_embedding_mae * loss_mark
        )
        loss_extraction = self.strength_extraction_mae * loss_mark

        return tf.reduce_mean(loss_embedding), tf.reduce_mean(loss_extraction)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "strength_embedding_mse": self.strength_embedding_mse,
            "strength_embedding_mae": self.strength_embedding_mae,
            "strength_extraction_mae": self.strength_extraction_mae,
        }
