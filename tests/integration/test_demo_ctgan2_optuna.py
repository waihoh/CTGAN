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

    study = optuna.create_study(direction="minimize")
    # TODO: timeout?
    study.optimize(objective, n_trials=50, timeout=600, callbacks=[callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save best mdl
    best_mdl.save(
        best_mdl.logger.dirpath + "/" + "ctgan_model_" + best_mdl.logger.PID + "_" + best_mdl.logger.dt.now().strftime(
            best_mdl.logger.datetimeformat) + ".pkl")

