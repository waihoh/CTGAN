# For testing of the three models
from ctgan import CTGANSynthesizer, TableganSynthesizer, TVAESynthesizer
from ctgan.transformer import DataTransformer
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy
from ctgan.sampler import Sampler
from ctgan.tvae import loss_function

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

# Save synthetic data and trained model
timestamp = model.logger.dt.now().strftime(model.logger.datetimeformat)

model_filepath = model.logger.dirpath + "/" + modelname + "_model_" + model.logger.PID + "_" + timestamp + ".pkl"
model.save(model_filepath)

# ------------------------------------------------------------------------------------
# Reload
# ------------------------------------------------------------------------------------
print("*" * 100)
print('RELOAD AND TRAIN')

# Reload a model
model2 = torch.load(model_filepath, map_location=torch.device('cpu'))  # cuda:1

# # Make changes to the toy data. Test with different data.
# data2 = pd.DataFrame({
#     'continuous1': np.random.random(3000)*3,
#     'discrete1': np.repeat([2, 1, 3], [2850, 75, 75]),
#     'discrete2': np.repeat(["b", "a", np.nan], [740, 2258, 2]),
#     'discrete3': np.repeat([7, 6], [300, 2700]),
# })

# Test with similar data.
data2 = pd.DataFrame({
    'continuous1': np.random.random(3000),
    'discrete1': np.repeat([1, 2, 3], [2850, 75, 75]),
    'discrete2': np.repeat(["a", "b", np.nan], [1740, 1258, 2]),
    'discrete3': np.repeat([6, 7], [300, 2700])
})

# Shuffle rows
data2 = data2.sample(frac=1).reset_index(drop=True)

# headers of discrete columns
discrete_columns2 = ['discrete1', 'discrete2', 'discrete3']

model2.fit(data2, discrete_columns, model_summary=False, trans="VGM", reload=True)

model2_filepath = model2.logger.dirpath + "/" + modelname + "_model2_" + model.logger.PID + "_" + timestamp + ".pkl"
model2.save(model2_filepath)
