# -*- coding: utf-8 -*-
"""
Set of functions to search hyper-parameters for watermark model.
"""
import json
import os
import keras
import tensorflow as tf
from typing import Tuple, Dict, Any
from glob import glob
from os.path import join, sep
from sklearn.model_selection import train_test_split

from src.addons.data.pipeline import train_pipeline, test_pipeline
from src.addons.learn.train import WatermarkTrainer


def get_dataset(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create the dataset for training and testing.

    Parameters
    ----------
    batch_size : int
        Batch size for dataset.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        Training and testing datasets.
    """
    images_path = glob(join(os.environ.get("RAW_PATH"), "images") + sep + "*.jpg")
    train_path, test_path = train_test_split(images_path, test_size=0.2, random_state=1335)
    return train_pipeline(train_path, batch=batch_size), test_pipeline(test_path, batch=batch_size)


def get_optimizer(optimizer: str, config: Dict[str, Any]) -> keras.optimizers.Optimizer:
    """
    Get the optimizer for training.

    Parameters
    ----------
    optimizer : str
        Optimizer name.
    config : Dict[str, Any]
        Configuration of optimizer.

    Returns
    -------
    keras.optimizers.Optimizer
        Optimizer instance.
    """
    optimizer = getattr(keras.optimizers, optimizer)(**config)
    return optimizer


def create_optimizer(trial, model_name: str) -> keras.optimizers.Optimizer:
    """
    Create an optimizer for model's training.

    Parameters
    ----------
    trial
    model_name : str
        Name of the model.

    Returns
    -------
    Optimizer
        Optimizer for training.
    """
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_selected = trial.suggest_categorical(f"optimizer_{model_name}", ["RMSprop", "Adam", "SGD"])
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(f"rmsprop_{model_name}_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["weight_decay"] = trial.suggest_float(f"rmsprop_{model_name}_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float(f"rmsprop_{model_name}_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float(f"adam_{model_name}_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(f"sgd_{model_name}_opt_learning_rate", 1e-5, 1e-1, log=True)
        kwargs["momentum"] = trial.suggest_float(f"sgd_opt_{model_name}_momentum", 1e-5, 1e-1, log=True)

    return get_optimizer(optimizer=optimizer_selected, config=kwargs)


def create_loss(trial) -> Dict[str, float]:
    """
    Create a loss function.

    Parameters
    ----------
    trial

    Returns
    -------
    Dict[str, Any]
        Loss configuration for training.
    """
    kwargs = {
        "strength_embedding_mse": trial.suggest_float("loss_strength_embedding_mse", 10, 100),
        "strength_embedding_mae": trial.suggest_float("loss_strength_embedding_mae", 0.1, 0.99, log=True),
        "strength_extraction_mae": trial.suggest_float("loss_strength_extraction_mae", 10, 100),
    }
    return kwargs


def objective_model(trial) -> Tuple[float, float]:
    trainer = WatermarkTrainer()
    trainer.compile(
        embedding_optimizer=create_optimizer(trial, model_name="watermark"),
        extractor_optimizer=create_optimizer(trial, model_name="extract"),
    )

    # ##: Create datasets.
    batch_size = trial.suggest_int("batch_size", 16, 64)
    train_set, test_set = get_dataset(batch_size=batch_size)
    loss = {"strength_embedding_mse": 45.0, "strength_embedding_mae": 0.2, "strength_extraction_mae": 20}

    # ##: Train model.
    epochs = trial.suggest_int("epochs", 30, 50)
    for epoch in range(epochs):
        print(f"Epoch n°{epoch+1}")
        trainer.learn(train_set=train_set, loss=loss)

    # ##: Test model.
    psnr, ber = trainer.evaluate(test_set, attack="identity")

    return psnr, ber


def objective_loss(trial) -> Tuple[float, float]:
    # ##: Get best hyper-parameters.
    with open(join(os.environ.get("PARAMS_PATH"), "best_params.json"), "r", encoding="utf-8") as file:
        best = json.load(file)

    trainer = WatermarkTrainer()
    trainer.compile(
        embedding_optimizer=get_optimizer(best["embedding"]["optimizer"], best["embedding"]["configuration"]),
        extractor_optimizer=get_optimizer(best["extract"]["optimizer"], best["extract"]["configuration"]),
    )

    # ##: Create datasets.
    train_set, test_set = get_dataset(batch_size=best["batch_size"])

    # ##: Train model.
    loss = create_loss(trial)
    for epoch in range(best["epochs"]):
        print(f"Epoch n°{epoch+1}:")
        trainer.learn(train_set=train_set, loss=loss)

    # ##: Test model.
    psnr, ber = trainer.evaluate(test_set, attack="identity")

    return psnr, ber
