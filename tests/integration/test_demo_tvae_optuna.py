# test the GitHub demo
import os
from ctgan import TVAESynthesizer
import pandas as pd
import torch
import numpy as np
import optuna
from ctgan.config import tvae_setting as cfg
from ctgan.logger import Logger

def sortlists(metrics, fns):
    metrics_sorted, fns_sorted = (list(t) for t in zip(*sorted(zip(metrics, fns))))
    return metrics_sorted, fns_sorted

cwd = os.getcwd()
print("Current working directory is:", cwd)

# logger to save optuna statistics
optuna_logger = Logger(filename="optuna_trials_summary.txt")
optuna_logger.change_dirpath(cwd + "/" + "TVAE_" + optuna_logger.PID)

# # using a toy example to test tablegan
data = pd.DataFrame({
    'continuous1': np.random.random(1000),
    'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
    'discrete2': np.repeat(["a", "b"], [580, 420]),
    'discrete3': np.repeat([6, 7], [100, 900])
})

# index of columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

# for saving the best model
# best_mdl = None
tvae_mdl = None
metric_vals = []
mdl_fns = []
num_max_mdls = 5

n_trials = 10

# generate unique seed for each trial
seed_list = np.arange(n_trials).tolist()
np.random.shuffle(seed_list)

def objective(trial):
    this_seed = seed_list[-1]
    seed_list.pop()

    torch.manual_seed(this_seed)
    np.random.seed(this_seed)

    cfg.LEARNING_RATE = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    # cfg.DEPTH = trial.suggest_int('depth', 2, 4)
    # cfg.WIDTH = trial.suggest_int('width', 128, 512, log=True)

    # initialize a new tvae model
    global tvae_mdl
    tvae_mdl = TVAESynthesizer()

    # Create a new folder to save the training results
    tvae_mdl.logger.change_dirpath(tvae_mdl.logger.dirpath + "/TVAE_" + tvae_mdl.logger.PID)  ## create a folder with PID

    # Record the seed number
    tvae_mdl.logger.write_to_file('Both seed number ' + str(this_seed))

    # NOTE: to use Optuna, pass trial to fit function
    tvae_mdl.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial)

    return tvae_mdl.prop_dis_validation


# saving the best model
# see reply by Toshihiko Yanase in https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study
def callback(study, trial):
    global metric_vals, mdl_fns, num_max_mdls #, best_mdl
    if trial.state == optuna.trial.TrialState.COMPLETE:
        this_model_fn = "TVAE_model_" + str(trial.number) + "_" \
                        + optuna_logger.PID + "_" \
                        + optuna_logger.dt.now().strftime(optuna_logger.datetimeformat)

        if len(metric_vals) < num_max_mdls:
            metric_vals.append(tvae_mdl.prop_dis_validation)
            mdl_fns.append(this_model_fn)
            metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

            tvae_mdl.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")
        else:
            print(mdl_fns)
            if tvae_mdl.prop_dis_validation < metric_vals[-1]:
                # remove the previously saved model
                metric_vals.pop()
                mdl_fn_discard = mdl_fns.pop()

                os.remove(optuna_logger.dirpath + "/" + mdl_fn_discard + ".pkl")

                # add the new record
                metric_vals.append(tvae_mdl.prop_dis_validation)
                mdl_fns.append(this_model_fn)
                metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                tvae_mdl.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")

    # if study.best_trial == trial:
    #     best_mdl = tvae_mdl


if __name__ == "__main__":
    cfg.EPOCHS = 20  # just to speed up the test

    # Remove/replace NopPruner if we want to use a pruner.
    # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
    # study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # # save best mdl. Use the same timestamp as its logger
    # mdl_fn = "tvae_model_" + best_mdl.logger.PID + "_" + best_mdl.logger.datetimeval + ".pkl"
    # best_mdl.save(best_mdl.logger.dirpath + "/" + mdl_fn)
    #
    # best_mdl.logger.write_to_file("Saved best model: " + mdl_fn)
    #
    # best_mdl.logger.write_to_file("Study statistics: ")
    # best_mdl.logger.write_to_file("  Number of finished trials: " + str(len(study.trials)))
    # best_mdl.logger.write_to_file("  Number of pruned trials: " + str(len(pruned_trials)))
    # best_mdl.logger.write_to_file("  Number of complete trials: " + str(len(complete_trials)))
    #
    # best_mdl.logger.write_to_file("Best trial:")
    # trial = study.best_trial
    #
    # best_mdl.logger.write_to_file("  Value: " + str(trial.value))
    #
    # best_mdl.logger.write_to_file("  Params: ")
    # for key, value in trial.params.items():
    #     best_mdl.logger.write_to_file("    {}: {}".format(key, value))

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
        samples = temp_mdl.sample(10, condition_column=None, condition_value=None)
        samples.to_csv(output_sample_path, index=False, header=True)
