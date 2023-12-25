# -*- coding: utf-8 -*-
"""
Script for training watermark models.
"""

if __name__ == "__main__":
    import os
    import json
    from os.path import join
    from datetime import datetime

    from dotenv import find_dotenv, load_dotenv

    from src.addons.learn.train import Watermark
    from src.addons.learn.search import get_optimizer, get_dataset

    load_dotenv(find_dotenv())

    # ##: Get best hyper-parameters.
    with open(join(os.environ.get("PARAMS_PATH"), "best_params.json"), "r", encoding="utf-8") as file:
        best = json.load(file)

    trainer = Watermark.create()
    trainer.compile(
        embedding_optimizer=get_optimizer(best["embedding"]["optimizer"], best["embedding"]["configuration"]),
        extractor_optimizer=get_optimizer(best["extract"]["optimizer"], best["extract"]["configuration"]),
    )

    # ##: Create datasets.
    train_set, test_set = get_dataset(batch_size=best["batch_size"])

    # ##: Train model.
    trainer.fit(train_set=train_set, epochs=best["epochs"], loss=best["loss"], early_stopping=False)

    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    trainer.embedding.save(join(os.environ.get("MODELS_PATH"), f"embedding.{now}.model.h5"))
    trainer.extractor.save(join(os.environ.get("MODELS_PATH"), f"extractor.{now}.model.h5"))
