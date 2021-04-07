# For testing of the three models with Optuna
import os
from ctgan import CTGANSynthesizer, TableganSynthesizer, TVAESynthesizer
import pandas as pd
import torch
import numpy as np
import optuna
from ctgan.logger import Logger

'''
USER INPUT IS REQUIRED HERE
- Change the modelname to test different models.
- select between ctgan, tablegan or tvae- Note that hyperparameters are in config.py.
'''
modelname = 'tvae'  # ctgan, tablegan, tvae

"""
Sample code
"""
# Seeding.
seednum = 0
torch.manual_seed(seednum)
np.random.seed(seednum)

# Create a toy example of 3000 rows for testing
data = pd.DataFrame({
    'continuous1': np.random.random(3000),
    'continuous2': np.random.random(3000),
    'discrete1': np.repeat([1, 2, 3], [2850, 75, 75]),
    'discrete2': np.repeat([4,5,np.nan], [1740, 1258, 2]),
    'discrete3': np.repeat([6, 7], [300, 2700])
})



# Shuffle rows
data = data.sample(frac=1).reset_index(drop=True)

# headers of discrete columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

# Create an Optuna logger to save optuna statistics
cwd = os.getcwd()
optuna_logger = Logger(filename="optuna_trials_summary.txt")
optuna_logger.change_dirpath(cwd + "/" + modelname + "_" + optuna_logger.PID)

# Save num_max_mdls, currently set as 5.
num_max_mdls = 5
model = None
metric_vals = []
mdl_fns = []
n_trials = 10

# For sorting of the list that saves names of top models, eg. top 5 models.
def sortlists(metrics, fns):
    metrics_sorted, fns_sorted = (list(t) for t in zip(*sorted(zip(metrics, fns))))
    return metrics_sorted, fns_sorted

# Optuna requires an objective function to return the metric for tracking/evaluation
# The settings of the models can be done in this function too.
def objective(trial):
    # initialize a new model
    global model
    if modelname == 'ctgan':
        model = CTGANSynthesizer()
        from ctgan.config import ctgan_setting as cfg
    elif modelname == 'tablegan':
        model = TableganSynthesizer()
        from ctgan.config import tablegan_setting as cfg
    elif modelname == 'tvae':
        model = TVAESynthesizer()
        from ctgan.config import tvae_setting as cfg
    else:
        ValueError('In valid modelname')

    # Changing learning rate in each trial
    cfg.LEARNING_RATE = trial.suggest_float("lr", 1e-6, 1e-3, log=True)

    # Create a new folder to save the training results
    model.logger.change_dirpath(model.logger.dirpath + "/" + modelname + "_" + model.logger.PID)  #

    # NOTE: to use Optuna, pass trial to fit function
    model.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial)

    return model.optuna_metric

# Save file results of each trial.
# Save the names of top models in this callback function.
def callback(study, trial):
    global metric_vals, mdl_fns, num_max_mdls
    if trial.state == optuna.trial.TrialState.COMPLETE:
        this_model_fn = modelname + "_model_" + str(trial.number) + "_" \
                        + optuna_logger.PID + "_" \
                        + optuna_logger.dt.now().strftime(optuna_logger.datetimeformat)

        if len(metric_vals) < num_max_mdls:
            metric_vals.append(model.optuna_metric)
            mdl_fns.append(this_model_fn)
            metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

            model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")
        else:
            if model.optuna_metric < metric_vals[-1]:
                # remove the previously saved model
                metric_vals.pop()
                mdl_fn_discard = mdl_fns.pop()

                os.remove(optuna_logger.dirpath + "/" + mdl_fn_discard + ".pkl")

                # add the new record
                metric_vals.append(model.optuna_metric)
                mdl_fns.append(this_model_fn)
                metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")

    ## Info: for saving the best model,
    ## see reply by Toshihiko Yanase in https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study
    # if study.best_trial == trial:
    #     best_mdl = model


if __name__ == "__main__":
    # Remove/replace NopPruner if we want to use a pruner.
    # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
    # study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Show and save Optuna statistics
    optuna_logger.write_to_file("Study statistics: ")
    optuna_logger.write_to_file("  Number of finished trials: " + str(len(study.trials)))
    optuna_logger.write_to_file("  Number of pruned trials: " + str(len(pruned_trials)))
    optuna_logger.write_to_file("  Number of complete trials: " + str(len(complete_trials)))

    optuna_logger.write_to_file("Best trial:")
    trial = study.best_trial

    optuna_logger.write_to_file("  Value: " + str(trial.value))

    optuna_logger.write_to_file("  Params: ")
    for key, value in trial.params.items():
        optuna_logger.write_to_file("    {}: {}".format(key, value))

    optuna_logger.write_to_file(study.trials_dataframe().
                                sort_values(by='value', ascending=True).
                                to_string(header=True, index=False))

    # Generate synthetic data with each save model
    for mdl_fn in mdl_fns:
        filepath = optuna_logger.dirpath + "/" + mdl_fn + ".pkl"
        sample_fn = mdl_fn + "_sample.csv"
        output_sample_path = os.path.join(optuna_logger.dirpath, sample_fn)
        temp_mdl = torch.load(filepath, map_location=torch.device('cpu'))
        samples = temp_mdl.sample(3000, condition_column=None, condition_value=None)
        samples.to_csv(output_sample_path, index=False, header=True)
