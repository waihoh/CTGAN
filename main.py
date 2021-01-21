import pandas as pd
import os
import time
import torch
import numpy as np

from ctgan.argparser import ParserOutput
from ctgan import CTGANSynthesizer
from ctgan import TableganSynthesizer
from ctgan import TVAESynthesizer

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

    # select model
    if parser.model_type == 'ctgan':
        model = CTGANSynthesizer()
    elif parser.model_type == 'tablegan':
        model = TableganSynthesizer()
    elif parser.model_type == 'tvae':
        model = TVAESynthesizer()
    else:
        ValueError('The selected model, ' + parser.model_type + ', is invalid.')

    # read the training data
    data = pd.read_csv(data_path)

    # read list of discrete variables
    with open(discrete_cols_path, "r+") as f:
        discrete_columns = f.read().splitlines()

    # update logger output path
    model.logger.change_dirpath(parser.outputdir)

    # Train the model
    start_time = time.time()
    model.fit(data, discrete_columns)
    elapsed_time = time.time() - start_time
    model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)

    # Save the model in the same folder as the log file
    model_fn = parser.model_type + "_" + model.logger.PID + "_" + model.logger.now.strftime(model.logger.datetimeformat) + ".pkl"
    output_model_path = os.path.join(model.logger.dirpath, model_fn)
    model.save(output_model_path)
