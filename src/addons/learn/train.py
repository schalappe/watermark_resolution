# -*- coding: utf-8 -*-
"""
Trains the neural network.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from keras.callbacks import EarlyStopping
from keras.optimizers import Optimizer
from tensorflow.data import Dataset

from src.addons.watermark.models import WatermarkModel


def train_model(
    dataset: Tuple[Dataset, Dataset], optimizers: Tuple[Optimizer, Optimizer], loss: Dict[str, Any], epochs: int
) -> Tuple[float, float]:
    """
    Create and train model.

    Parameters
    ----------
    dataset : Tuple[Dataset, Dataset]
        Dataset used for training and evaluation.
    optimizers : Tuple[Optimizer, Optimizer]
        Optimizer used for training.
    loss : Dict[str, Any]
        Dictionary of loss arguments.
    epochs : int
        Number of training epochs.

    Returns
    -------
    Tuple[float, float]
        Peak signal noise ratio and bit errors ratio on test set.
    """
    # ##: Create and compile model.
    models = WatermarkModel.from_ashes(image_dims=(128, 128, 3), mark_dims=(8, 8, 1))

    # ##: Compile model.
    embedding_optimizer, extractor_optimizer = optimizers
    models.compile(
        embedding_optimizer=embedding_optimizer,
        extractor_optimizer=extractor_optimizer,
        loss=loss,
    )

    # ##: Train model.
    train_ds, test_ds = dataset
    models.fit(
        train_ds,
        epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="loss_embedding", patience=5),
            EarlyStopping(monitor="loss_extractor", patience=5),
        ],
    )

    # ##: Test model.
    return models.evaluate(test_ds, attack="identity")
