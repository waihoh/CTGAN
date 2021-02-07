import os
import time
import torch
import numpy as np
import pandas as pd

from ctgan.argparser import ParserOutput
from ctgan.logger import Logger
from ctgan import CTGANSynthesizer
from ctgan import TableganSynthesizer
from ctgan import TVAESynthesizer

import optuna
from ctgan import config as cfg
# import pickle

'''
Run hyper-parameter tuning using Optuna
'''
# update inputs
parser = ParserOutput()

# logger to save optuna statistics
optuna_logger = Logger(filename="optuna_trials_summary.txt")
optuna_logger.change_dirpath(parser.outputdir + "/" + parser.model_type + "_" + optuna_logger.PID)

# function to sort two lists together

def sortlists(metrics, fns):
    metrics_sorted, fns_sorted = (list(t) for t in zip(*sorted(zip(metrics, fns))))
    return metrics_sorted, fns_sorted

if parser.proceed:
    # get paths
    data_path = os.path.join(parser.datadir, parser.data_fn)
    discrete_cols_path = os.path.join(parser.datadir, parser.discrete_fn)
    if not os.path.isfile(data_path):
        ValueError('Training data file ' + data_path + " does not exists.")
    if not os.path.isfile(discrete_cols_path):
        ValueError('Discrete text file ' + discrete_cols_path + " does not exists.")

    # generate unique seed for each trial
    seed_list = np.arange(parser.trials).tolist()
    np.random.shuffle(seed_list)

    # model to be trained
    model = None

    # to record the top num_max_mdls models
    metric_vals = []
    mdl_fns = []
    num_max_mdls = parser.trials

    def objective(trial):
        global model

        # get the seed number from the last element of seed_list and remove it from the list.
        this_seed = seed_list[-1]
        seed_list.pop()
        torch.manual_seed(this_seed)
        np.random.seed(this_seed)


        if parser.model_type == 'ctgan':
            if trial is not None:
                cfg.ctgan_setting.GENERATOR_LEARNING_RATE = trial.suggest_float('ct_gen_lr', 2e-6, 2e-4, log=True)
                cfg.ctgan_setting.DISCRIMINATOR_LEARNING_RATE = trial.suggest_float('ct_dis_lr', 2e-6, 2e-4, log=True)
                cfg.ctgan_setting.EPOCHS = trial.suggest_int('ct_epochs',300,900,step=100)
                cfg.ctgan_setting.BATCH_SIZE = trial.suggest_int('ct_batchsize',500,1000,step=100)
                # initialize a new model
                model = CTGANSynthesizer()

        elif parser.model_type == 'tablegan':
            if trial is not None:
                cfg.tablegan_setting.LEARNING_RATE = trial.suggest_float('tbl_lr', 2e-6, 2e-4, log=True)
                cfg.tablegan_setting.EPOCHS = trial.suggest_int('tbl_epochs',30,120,step=30)
                cfg.tablegan_setting.BATCH_SIZE = trial.suggest_int('tbl_batchsize', 500, 1000, step=100)
                # initialize a new model
                model = TableganSynthesizer()

        elif parser.model_type == 'tvae':
            if trial is not None:
                cfg.tvae_setting.LEARNING_RATE = trial.suggest_float('tv_lr', 1e-4, 1e-2, log=True)
                cfg.tvae_setting.EPOCHS = trial.suggest_int('tv_epochs',300,900,step=100)
                cfg.tvae_setting.BATCH_SIZE = trial.suggest_int('tv_batchsize', 500, 1000, step=100)
                cfg.tvae_setting.DEPTH = trial.suggest_categorical('tv_depth',[2,3])
                cfg.tvae_setting.CONDGEN = trial.suggest_categorical('tv_condgen',[True,False])
                # initialize a new model
                model = TVAESynthesizer()

        else:
            ValueError('The selected model, ' + parser.model_type + ', is invalid.')

        # Create a new folder to save the training results
        model.logger.change_dirpath(optuna_logger.dirpath)  ## create a folder with PID

        # Record the seed number
        model.logger.write_to_file('Both seed number ' + str(this_seed))

        data = pd.read_csv(data_path).values

        # read list of discrete variables
        with open(discrete_cols_path, "r+") as f:
            discrete_columns = f.read().splitlines()

        # Train the model
        start_time = time.time()

        # model.fit(data, discrete_columns)
        model.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial,
                  transformer=parser.transformer, in_val_data=parser.val_data_fn,
                  threshold=parser.threshold)

        elapsed_time = time.time() - start_time
        model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)

        # if model.prop_dis_validation <= 0.5:
        #     with open(model.logger.dirpath+"/model{}.pkl".format(trial.number), "wb") as fout:
        #         pickle.dump(model, fout)

        return model.prop_dis_validation


    def callback(study, trial):
        global metric_vals, mdl_fns, num_max_mdls  # , best_mdl
        if trial.state == optuna.trial.TrialState.COMPLETE:
            this_model_fn = parser.model_type + "_model_" \
                            + str(trial.number) + "_" \
                            + optuna_logger.PID + "_" \
                            + optuna_logger.dt.now().strftime(optuna_logger.datetimeformat)

            if len(metric_vals) < num_max_mdls:
                metric_vals.append(model.prop_dis_validation)
                mdl_fns.append(this_model_fn)
                metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")
            else:
                print(mdl_fns)
                if model.prop_dis_validation < metric_vals[-1]:
                    # remove the previously saved model
                    metric_vals.pop()
                    mdl_fn_discard = mdl_fns.pop()

                    os.remove(optuna_logger.dirpath + "/" + mdl_fn_discard + ".pkl")

                    # add the new record
                    metric_vals.append(model.prop_dis_validation)
                    mdl_fns.append(this_model_fn)
                    metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                    model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")

        # if study.best_trial == trial:
        #     best_mdl = tvae_mdl

if __name__ == "__main__":
    # Remove/replace NopPruner if we want to use a pruner.
    # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=parser.trials, callbacks=[callback])
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # write training statistics to log file
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
        samples = temp_mdl.sample(parser.samplesize, condition_column=None, condition_value=None)
        samples.to_csv(output_sample_path, index=False, header=True)
