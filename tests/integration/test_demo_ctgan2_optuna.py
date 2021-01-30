# test the GitHub demo

# based on Optuna example in https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py
import os
from ctgan.synthesizer2 import CTGANSynthesizer2
import pandas as pd
import numpy as np
import optuna
from ctgan.config import ctgan_setting as cfg

cwd = os.getcwd()
print("Current working directory is:", cwd)

# using a toy example to test tablegan
data = pd.DataFrame({
    'continuous1': np.random.random(1000),
    'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
    'discrete2': np.repeat(["a", "b"], [580, 420]),
    'discrete3': np.repeat([6, 7], [100, 900])
})

# index of columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

# for saving the best model
best_mdl = None
ctgan_mdl = None


def objective(trial):
    cfg.GENERATOR_LEARNING_RATE = trial.suggest_float("gen_lr", 1e-6, 1e-3, log=True)
    cfg.DISCRIMINATOR_LEARNING_RATE = trial.suggest_float("dis_lr", 1e-6, 1e-3, log=True)

    # initialize a new ctgan model
    global ctgan_mdl
    ctgan_mdl = CTGANSynthesizer2()

    # NOTE: to use Optuna, pass trial to fit function
    ctgan_mdl.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial)

    return ctgan_mdl.val_metric


# saving the best model
# see reply by Toshihiko Yanase in https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study
def callback(study, trial):
    global best_mdl
    if study.best_trial == trial:
        best_mdl = ctgan_mdl


if __name__ == "__main__":
    cfg.EPOCHS = 20  # just to speed up the test

    # Remove/replace NopPruner if we want to use a pruner.
    # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=10, callbacks=[callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # save best mdl. Use the same timestamp as its logger
    mdl_fn = "ctgan_model_" + best_mdl.logger.PID + "_" + best_mdl.logger.datetimeval + ".pkl"
    best_mdl.save(best_mdl.logger.dirpath + "/" + mdl_fn)

    best_mdl.logger.write_to_file("Saved best model: " + mdl_fn)

    best_mdl.logger.write_to_file("Study statistics: ")
    best_mdl.logger.write_to_file("  Number of finished trials: " + str(len(study.trials)))
    best_mdl.logger.write_to_file("  Number of pruned trials: " + str(len(pruned_trials)))
    best_mdl.logger.write_to_file("  Number of complete trials: " + str(len(complete_trials)))

    best_mdl.logger.write_to_file("Best trial:")
    trial = study.best_trial

    best_mdl.logger.write_to_file("  Value: " + str(trial.value))

    best_mdl.logger.write_to_file("  Params: ")
    for key, value in trial.params.items():
        best_mdl.logger.write_to_file("    {}: {}".format(key, value))

