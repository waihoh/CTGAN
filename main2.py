import os
import time
import torch
import numpy as np
import pandas as pd

from ctgan.argparser import ParserOutput
from ctgan import CTGANSynthesizer
from ctgan import TableganSynthesizer
from ctgan import TVAESynthesizer

import optuna
from ctgan import config as cfg
import pickle


'''
Run parser function to update inputs by the user.
'''
# update inputs
parser = ParserOutput()

# Initialize seed
torch.manual_seed(parser.torch_seed)
np.random.seed(parser.numpy_seed)




if parser.proceed:
    # get paths
    data_path = os.path.join(parser.datadir, parser.data_fn)
    discrete_cols_path = os.path.join(parser.datadir, parser.discrete_fn)
    if not os.path.isfile(data_path):
        ValueError('Training data file ' + data_path + " does not exists.")
    if not os.path.isfile(discrete_cols_path):
        ValueError('Discrete text file ' + discrete_cols_path + " does not exists.")
    model = None

    def objective(trial):
        global model
        if parser.model_type == 'ctgan':
            if trial is not None:
                cfg.ctgan_setting.GENERATOR_LEARNING_RATE = trial.suggest_float('ct_gen_lr', 2e-6, 2e-4, log=True)
                cfg.ctgan_setting.DISCRIMINATOR_LEARNING_RATE = trial.suggest_float('ct_dis_lr', 2e-6, 2e-4, log=True)
                cfg.ctgan_setting.EPOCHS = trial.suggest_int('ct_epochs',300,900,step=100)
                cfg.ctgan_setting.BATCH_SIZE = trial.suggest_int('ct_batchsize',500,1000,step=100)
                # initialize a new model
                #global model
                model = CTGANSynthesizer()
            else:
                model = CTGANSynthesizer()
        elif parser.model_type == 'tablegan':
            if trial is not None:
                cfg.tablegan_setting.LEARNING_RATE = trial.suggest_float('tbl_lr', 2e-6, 2e-4, log=True)
                cfg.tablegan_setting.EPOCHS = trial.suggest_int('tbl_epochs',30,120,step=30)
                cfg.tablegan_setting.BATCH_SIZE = trial.suggest_int('tbl_batchsize', 500, 1000, step=100)
                # initialize a new model
                #global model
                model = TableganSynthesizer()
            else:
                model = TableganSynthesizer()
        elif parser.model_type == 'tvae':
            if trial is not None:
                cfg.tvae_setting.LEARNING_RATE = trial.suggest_float('tv_lr', 1e-4, 1e-2, log=True)
                cfg.tvae_setting.EPOCHS = trial.suggest_int('tv_epochs',300,900,step=100)
                cfg.tvae_setting.BATCH_SIZE = trial.suggest_int('tv_batchsize', 500, 1000, step=100)
                cfg.tvae_setting.DEPTH = trial.suggest_categorical('tv_depth',[2,3])
                cfg.tvae_setting.CONDGEN = trial.suggest_categorical('tv_condgen',[True,False])
                # initialize a new model
                #global model
                model = TVAESynthesizer()
            else:
                model = TVAESynthesizer()
        else:
            ValueError('The selected model, ' + parser.model_type + ', is invalid.')

        data = pd.read_csv(data_path).values

        # read list of discrete variables
        with open(discrete_cols_path, "r+") as f:
            discrete_columns = f.read().splitlines()

        # update logger output path
        model.logger.change_dirpath(parser.outputdir)

        # Train the model
        start_time = time.time()
        # model.fit(data, discrete_columns)
        model.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial,
                  transformer = parser.transformer, in_val_data=parser.val_data_fn, threshold=parser.threshold)
        if model.prop_dis_validation <= 0.5:
            with open(model.logger.dirpath+"/model{}.pkl".format(trial.number), "wb") as fout:
                pickle.dump(model, fout)
        elapsed_time = time.time() - start_time
        model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)
        return model.prop_dis_validation


    def callback(study, trial):
        global best_mdl
        if study.best_trial == trial:
            best_mdl = model

    if __name__ == "__main__":
        # Remove/replace NopPruner if we want to use a pruner.
        # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
        study.optimize(objective, n_trials=parser.trials, callbacks=[callback])
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        best_mdl.logger.write_to_file("Study statistics: ")
        best_mdl.logger.write_to_file("  Number of finished trials: " + str(len(study.trials)))
        best_mdl.logger.write_to_file("  Number of pruned trials: " + str(len(pruned_trials)))
        best_mdl.logger.write_to_file("  Number of complete trials: " +str(len(complete_trials)))

        best_mdl.logger.write_to_file("Best trial:")
        trial = study.best_trial

        best_mdl.logger.write_to_file("  Value: " + str(trial.value))

        best_mdl.logger.write_to_file("  Params: ")
        for key, value in trial.params.items():
            best_mdl.logger.write_to_file("    {}: {}".format(key, value))

        # save best mdl
        best_mdl.save(
            best_mdl.logger.dirpath + "/" + parser.model_type +"_model_" + best_mdl.logger.PID + "_" + best_mdl.logger.dt.now().strftime(
                best_mdl.logger.datetimeformat) + ".pkl")

        with open(best_mdl.logger.dirpath +"/trials_summary.txt" , "a") as myfile:
            myfile.write(study.trials_dataframe().sort_values(by='value',ascending=True).to_string(header=True,index=False))


