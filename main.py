import pandas as pd
import os
import time
import torch
import numpy as np

# Initialize seed
torch.manual_seed(0)
np.random.seed(0)

from ctgan.argparser import parser_func
from ctgan import CTGANSynthesizer
from ctgan import TableganSynthesizer
from ctgan import TVAESynthesizer

'''
Run parser function to update inputs by the user.
'''

# update inputs
model_type, datadir, outputdir, data_fn, discrete_fn = parser_func()

# get paths
data_path = os.path.join(datadir, data_fn)
discrete_cols_path = os.path.join(datadir, discrete_fn)

# select model
if model_type == 'ctgan':
    model = CTGANSynthesizer()
elif model_type == 'tablegan':
    model = TableganSynthesizer()
elif model_type == 'tvae':
    model = TVAESynthesizer()
else:
    ValueError('The selected model, ' + model_type + ', is invalid.')

# read the training data
data = pd.read_csv(data_path)

# read list of discrete variables
with open(discrete_cols_path, "r+") as f:
    discrete_columns = f.read().splitlines()

# update logger output path
model.logger.change_dirpath(outputdir)

# Train the model
start_time = time.time()
model.fit(data, discrete_columns)
elapsed_time = time.time() - start_time
model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)

# Save the model in the same folder as the log file
model_fn = model_type + "_" + model.logger.PID + "_" + model.logger.now.strftime(model.logger.datetimeformat) + ".pkl"
output_model_path = os.path.join(model.logger.dirpath, model_fn)
model.save(output_model_path)
