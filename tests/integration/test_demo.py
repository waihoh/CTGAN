# For testing of the three models
from ctgan import CTGANSynthesizer, TableganSynthesizer, TVAESynthesizer
import pandas as pd
import numpy as np
import torch

'''
USER INPUT IS REQUIRED HERE
- Change the modelname to test different models.
- select between ctgan, tablegan or tvae
- Note that hyperparameters are in config.py.
'''
modelname = 'ctgan'  # ctgan, tablegan, tvae

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
    'discrete1': np.repeat([1, 2, 3], [2850, 75, 75]),
    'discrete2': np.repeat(["a", "b", np.nan], [1740, 1258, 2]),
    'discrete3': np.repeat([6, 7], [300, 2700])
})

# Shuffle rows
data = data.sample(frac=1).reset_index(drop=True)

# headers of discrete columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

if modelname == 'ctgan':
    model = CTGANSynthesizer()
elif modelname == 'tablegan':
    model = TableganSynthesizer()
elif modelname == 'tvae':
    model = TVAESynthesizer()
else:
    ValueError('In valid modelname')

# Create a folder with PID to store output results
model.logger.change_dirpath(model.logger.dirpath + "/" + modelname + "_" + model.logger.PID)

# Train the model
model.fit(data, discrete_columns, model_summary=False, trans="VGM")

# Generate synthetic data of size 10
samples = model.sample(10)

# Save synthetic data and trained model
timestamp = model.logger.dt.now().strftime(model.logger.datetimeformat)

# 1) synthetic data
sample_filepath = model.logger.dirpath + "/" + modelname + "_samples_" + model.logger.PID + "_" + timestamp + ".csv"
samples.to_csv(sample_filepath, index=False, header=True)

# 2) trained model
model_filepath = model.logger.dirpath + "/" + modelname + "_model_" + model.logger.PID + "_" + timestamp + ".pkl"
model.save(model_filepath)
