# -*- coding: utf-8 -*-
"""
Script for training watermark models.
"""
import os
from glob import glob
from os.path import join, sep

from tensorflow.data import Dataset

from src.addons.data.pipeline import train_pipeline
from src.addons.learn.search import get_optimizer
from src.addons.watermark.models import WatermarkModel


def get_dataset(batch_size: int) -> Dataset:
    """
    Create the dataset for training.

    Parameters
    ----------
    batch_size : int
        Batch size for dataset.

    Returns
    -------
    Dataset
        Training datasets.
    """
    images_path = glob(join(os.environ.get("RAW_PATH"), "train") + sep + "*.jpg")
    return train_pipeline(images_path, batch=batch_size)


if __name__ == "__main__":
    import json

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    # ##: Get best hyper-parameters.
    with open(join(os.environ.get("MODELS_PATH"), "best_params.json"), "r", encoding="utf-8") as file:
        best = json.load(file)

    # ##: Create model and compile.
    models = WatermarkModel.from_ashes(image_dims=(128, 128, 3), mark_dims=(8, 8, 1))
    models.compile(
        embedding_optimizer=get_optimizer(best["embedding"]["optimizer"], best["embedding"]["configuration"]),
        extractor_optimizer=get_optimizer(best["extract"]["optimizer"], best["extract"]["configuration"]),
        loss=best["loss"],
    )

    # ##: Create datasets.
    train_set = get_dataset(batch_size=best["batch_size"])

    # ##: Train model.
    models.fit(train_set, epochs=best["epochs"])
    models.save(join(os.environ.get("MODELS_PATH"), "storage"))
