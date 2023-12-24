# -*- coding: utf-8 -*-
"""
Search best parameters for watermark neural network.
"""

if __name__ == "__main__":
    import os
    from datetime import datetime

    import optuna
    from dotenv import find_dotenv, load_dotenv

    from src.addons.learn.search import objective_model
    from src.addons.visualize.table import print_tables

    load_dotenv(find_dotenv())

    study_name = f"model_params_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    storage_name = f"sqlite:///{os.environ.get('PARAMS_PATH')}/watermark.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize", "minimize"])
    study.optimize(objective_model, n_trials=10)

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_psnr = max(study.best_trials, key=lambda t: t.values[0])
    print("Trial with highest PSNR: ")
    print_tables(
        "Trial with highest PSNR",
        headers=tuple(trial_with_highest_psnr.params.keys()),
        contents=[tuple(trial_with_highest_psnr.params.values())],
    )
