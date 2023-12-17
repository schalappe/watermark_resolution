# -*- coding: utf-8 -*-
"""
Trains the neural network.
"""
import os
import keras
import tensorflow as tf
from typing import Tuple, Dict, Any
from glob import glob
from os.path import join, sep
from sklearn.model_selection import train_test_split

from src.addons.watermark.loss import watermark_loss, mean_squared_error, mean_absolute_error
from src.addons.watermark.models import create_watermark, create_extract_mark
from src.addons.watermark.metric import peak_signal_noise_ratio, bit_error_ratio
from src.addons.augmenters.augment import random_attacks, attacks
from src.addons.data.pipeline import train_pipeline, test_pipeline


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
        "strength_embedding_mse": trial.suggest_float("loss_strength_embedding_mse", 1e-5, 1e-1, log=True),
        "strength_embedding_mae": trial.suggest_float("loss_strength_embedding_mae"),
        "strength_extraction_mae": trial.suggest_float("loss_strength_extraction_mae"),
    }
    return kwargs


@tf.function
def train_step(
    models: Tuple[keras.Model, keras.Model],
    optimizers: Tuple[keras.optimizers.Optimizer, keras.optimizers.Optimizer],
    data: Tuple[tf.Tensor, tf.Tensor],
    loss: Dict[str, Any],
) -> Dict[str, tf.Tensor]:
    """
    Single training step.

    Parameters
    ----------
    models : Tuple[keras.Model, keras.Model]
        Watermark and extract models used for training.
    optimizers : Tuple[keras.optimizers.Optimizer, keras.optimizers.Optimizer]
        Optimizers to use for training.
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
    watermark, extract_mark = models
    watermark_optimizer, extract_mark_optimizer = optimizers
    images, marks = data

    with tf.GradientTape() as watermark_tape, tf.GradientTape() as extract_tape:
        # ##: Embedding and marks.
        outputs = watermark([images, marks], training=True)
        attack_outputs = random_attacks(outputs)
        extracted_marks = extract_mark(attack_outputs, training=True)

        # ##: Calculates loss.
        loss_embedding, loss_extraction = watermark_loss((images, marks), (outputs, extracted_marks), **loss)

    # ##: computes gradient.
    grads_embedding = watermark_tape.gradient(loss_embedding, watermark.trainable_variables)
    grads_extraction = extract_tape.gradient(loss_extraction, extract_mark.trainable_variables)

    # ##: train extraction model.
    watermark_optimizer.apply_gradients(zip(grads_embedding, watermark.trainable_variables))
    extract_mark_optimizer.apply_gradients(zip(grads_extraction, extract_mark.trainable_variables))

    # ##: Compute metrics and loss.
    results = {
        "loss_embedding": loss_embedding,
        "loss_extract": loss_extraction,
        "PSNR": peak_signal_noise_ratio(images, outputs),
        "BER": bit_error_ratio(marks, extracted_marks),
    }

    return results


def learn(
    models: Tuple[keras.Model, keras.Model],
    optimizers: Tuple[keras.optimizers.Optimizer, keras.optimizers.Optimizer],
    train_set: tf.data.Dataset,
    loss: Dict[str, float],
) -> Tuple[float, float]:
    """
    Train a Watermark neural network on a dataset.

    Parameters
    ----------
    models : Tuple[keras.Model, keras.Model]
        Watermark neural network and extraction neural network to train.
    optimizers : Tuple[keras.optimizers.Optimizer, keras.optimizers.Optimizer]
        Optimizers to use for training.
    train_set : tf.data.Dataset
        Training dataset.
    loss : Dict[str, float]
        Loss configuration.

    Returns
    -------
    Tuple[float, float]
        Peak signal noise ratio and Bit error ratio.
    """

    # ##: Tracker for metric and loss.
    loss_embedding_tracker = keras.metrics.Mean(name="loss_embedding")
    loss_extract_tracker = keras.metrics.Mean(name="loss_extract")
    psnr_tracker = keras.metrics.Mean(name="peak_signal_to_noise_ratio")
    ber_tracker = keras.metrics.Mean(name="bit_error_ratio")

    # ##: Loop of train
    for batch, (images, marks) in enumerate(train_set):
        train_results = train_step(models=models, optimizers=optimizers, data=(images, marks), loss=loss)

        # ##. update metrics
        loss_embedding_tracker.update_state(train_results["loss_embedding"])
        loss_extract_tracker.update_state(train_results["loss_extract"])
        psnr_tracker.update_state(train_results["PSNR"])
        ber_tracker.update_state(train_results["BER"])

        # ##: Print train result.
        print(
            f"Batch n°{batch+1} -> "
            f"Loss watermark: {float(loss_embedding_tracker.result()):.4f} - "
            f"Loss extract: {float(loss_extract_tracker.result()):.4f} - "
            f"Peak signal noise ratio: {float(psnr_tracker.result()):.4f} - "
            f"Bit error ratio: {float(ber_tracker.result()):.4f}"
        )
    return psnr_tracker.result(), ber_tracker.result()


@tf.function
def evaluate(models: Tuple[keras.Model, keras.Model], test_set: tf.data.Dataset, attack: str) -> Tuple[float, float]:
    """
    Evaluate the model.

    Parameters
    ----------
    models : Tuple[keras.Model, keras.Model]
        Watermark neural network and extraction neural network to train.
    test_set : tf.data.Dataset
        Testing dataset.
    attack : str
        Attack to apply to watermark.

    Returns
    -------
    Tuple[float, float]
        Peak signal noise ratio and Bit error ratio.
    """
    watermark, extract_mark = models

    # ##: Tracker for metric.
    psnr_tracker = keras.metrics.Mean(name="peak_signal_to_noise_ratio")
    ber_tracker = keras.metrics.Mean(name="bit_error_ratio")

    for images, marks in test_set:
        # ##: Predicted.
        outputs = watermark([images, marks], training=False)
        attack_outputs = tf.cast(attacks[attack](outputs), tf.uint8)
        extracted_marks = extract_mark(attack_outputs, training=False)

        # ##: Metrics.
        psnr_tracker.update_state(peak_signal_noise_ratio(images, outputs))
        ber_tracker.update_state(bit_error_ratio(marks, extracted_marks))

    return psnr_tracker.result(), ber_tracker.result()


def objective_1(trial) -> Tuple[float, float]:
    watermark = create_watermark(image_dims=(512, 512, 3), mark_dims=(32, 32, 1), strength=1.0)
    extract_mark = create_extract_mark(image_dims=(512, 512, 3))

    # ##: Compile model with optimizer.
    watermark_optimizer = create_optimizer(trial, model_name="watermark")
    extract_mark_optimizer = create_optimizer(trial, model_name="extract")

    # ##: Create datasets.
    batch_size = trial.suggest_int("batch_size", 10, 32)
    train_set, test_set = get_dataset(batch_size=batch_size)
    loss = {"strength_embedding_mse": 45.0, "strength_embedding_mae": 0.2, "strength_extraction_mae": 20}

    # ##: Train model.
    epochs = trial.suggest_int("epochs", 100, 4000)
    for epoch in range(epochs):
        print(f"Epoch n°{epoch+1}")
        learn(
            models=(watermark, extract_mark),
            optimizers=(watermark_optimizer, extract_mark_optimizer),
            train_set=train_set,
            loss=loss,
        )

    # ##: Test model.
    psnr, ber = evaluate((watermark, extract_mark), test_set, attack="identity")

    return psnr, ber


def objective_2(trial, configs: Dict[str, Any]) -> Tuple[float, float]:
    watermark = create_watermark(image_dims=(512, 512, 3), mark_dims=(32, 32, 1), strength=1.0)
    extract_mark = create_extract_mark(image_dims=(512, 512, 3))

    # ##: Compile model with optimizer.
    watermark_optimizer = get_optimizer(configs["watermark"]["optimizer"], configs["watermark"]["configuration"])
    extract_mark_optimizer = get_optimizer(configs["extract"]["optimizer"], configs["extract"]["configuration"])

    # ##: Create datasets.
    train_set, test_set = get_dataset(batch_size=configs["batch_size"])

    # ##: Train model.
    loss = create_loss(trial)
    for epoch in range(configs["epochs"]):
        print(f"Epoch n°{epoch+1}:")
        learn(
            models=(watermark, extract_mark),
            optimizers=(watermark_optimizer, extract_mark_optimizer),
            train_set=train_set,
            loss=loss,
        )

    # ##: Test model.
    psnr, ber = evaluate((watermark, extract_mark), test_set, attack="identity")

    return psnr, ber
