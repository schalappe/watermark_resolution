# -*- coding: utf-8 -*-
"""
Search best parameters for watermark neural network.
"""

if __name__ == "__main__":
    import optuna
    from src.addons.watermark.train import objective_1
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective_1, n_trials=1)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
