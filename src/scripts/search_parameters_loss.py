# -*- coding: utf-8 -*-
"""
Search best parameters for watermark loss.
"""

if __name__ == "__main__":
    import os
    from datetime import datetime

    import optuna
    from dotenv import find_dotenv, load_dotenv

    from src.addons.learn.search import objective_loss
    from src.addons.visualize.table import print_best_params

    load_dotenv(find_dotenv())

    study_name = f"loss_params_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    storage_name = f"sqlite:///{os.environ.get('MODELS_PATH')}/params/watermark.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize", "minimize"])
    study.optimize(objective_loss, n_trials=30)
    print_best_params(study=study)
