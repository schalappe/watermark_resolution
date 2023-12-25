# -*- coding: utf-8 -*-
"""
Set of functions to search hyper-parameters for watermark model.
"""
import json
import os
from glob import glob
from os.path import join, sep
from typing import Any, Dict, Tuple

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.addons.data.pipeline import test_pipeline, train_pipeline
from src.addons.learn.train import train_model


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
    images_path = glob(join(os.environ.get("RAW_PATH"), "train") + sep + "*.jpg")
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
    optimizer = getattr(keras.optimizers.legacy, optimizer)(**config)
    return optimizer


def create_optimizer(trial, model_name: str, config: Dict[str, Any]) -> keras.optimizers.Optimizer:
    """
    Create an optimizer for model's training.

    Parameters
    ----------
    trial
    model_name : str
        Name of the model.
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Optimizer
        Optimizer for training.
    """
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_selected = trial.suggest_categorical(f"optimizer_{model_name}", config["optimizers"])
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(f"rmsprop_{model_name}_learning_rate", **config["learning_rate"])
        kwargs["weight_decay"] = trial.suggest_float(f"rmsprop_{model_name}_weight_decay", **config["weight_decay"])
        kwargs["momentum"] = trial.suggest_float(f"rmsprop_{model_name}_momentum", **config["momentum"])
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float(f"adam_{model_name}_learning_rate", **config["learning_rate"])
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(f"sgd_{model_name}_opt_learning_rate", **config["learning_rate"])
        kwargs["momentum"] = trial.suggest_float(f"sgd_opt_{model_name}_momentum", **config["momentum"])

    return get_optimizer(optimizer=optimizer_selected, config=kwargs)


def create_loss(trial, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Create a loss function.

    Parameters
    ----------
    trial
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Loss configuration for training.
    """
    kwargs = {
        "strength_embedding_mse": trial.suggest_float("loss_strength_embedding_mse", **config["loss_alpha"]),
        "strength_embedding_mae": trial.suggest_float("loss_strength_embedding_mae", **config["loss_beta"]),
        "strength_extraction_mae": trial.suggest_float("loss_strength_extraction_mae", **config["loss_alpha"]),
    }
    return kwargs


def objective_model(trial) -> Tuple[float, float]:
    """
    Optimize function for search model hyperparameters.

    Returns
    -------
    Tuple[float, float]
        Peak signal noise ratio and bit errors ratio
    """
    # ##: Get best hyper-parameters.
    with open(join(os.environ.get("MODELS_PATH"), "search_space.json"), "r", encoding="utf-8") as file:
        search = json.load(file)

    # ##: Create optimizers.
    embedding_optimizer = create_optimizer(trial, model_name="watermark", config=search["optimizer"])
    extractor_optimizer = create_optimizer(trial, model_name="extract", config=search["optimizer"])

    # ##: Create datasets.
    batch_size = trial.suggest_int("batch_size", **search["batch_size"])
    train_set, test_set = get_dataset(batch_size=batch_size)

    # ##: Epochs and loss arguments.
    epochs = trial.suggest_int("epochs", **search["epochs"])
    loss = {"strength_embedding_mse": 45.0, "strength_embedding_mae": 0.2, "strength_extraction_mae": 20}

    return train_model(
        dataset=(train_set, test_set), optimizers=(embedding_optimizer, extractor_optimizer), loss=loss, epochs=epochs
    )


def objective_loss(trial) -> Tuple[float, float]:
    """
    Optimize function for search loss arguments.

    Returns
    -------
    Tuple[float, float]
        Peak signal noise ratio and bit errors ratio
    """
    # ##: Get best hyper-parameters.
    with open(join(os.environ.get("MODELS_PATH"), "best_params.json"), "r", encoding="utf-8") as file:
        best = json.load(file)

    with open(join(os.environ.get("MODELS_PATH"), "search_space.json"), "r", encoding="utf-8") as file:
        search = json.load(file)

    # ##: Create optimizers.
    embedding_optimizer = get_optimizer(best["embedding"]["optimizer"], best["embedding"]["configuration"])
    extractor_optimizer = get_optimizer(best["extract"]["optimizer"], best["extract"]["configuration"])

    # ##: Create datasets.
    train_set, test_set = get_dataset(batch_size=best["batch_size"])

    # ##: Create loss arguments.
    loss = create_loss(trial, search["loss"])

    return train_model(
        dataset=(train_set, test_set),
        optimizers=(embedding_optimizer, extractor_optimizer),
        loss=loss,
        epochs=best["epochs"],
    )
