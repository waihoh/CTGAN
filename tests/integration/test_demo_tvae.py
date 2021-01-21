# test the GitHub demo
import os
from ctgan import load_demo
from ctgan import TVAESynthesizer
import pandas as pd
import numpy as np

cwd = os.getcwd()
print("Current working directory is:", cwd)

# # using a toy example to test tablegan
data = pd.DataFrame({
    'continuous1': np.random.random(1000),
    'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
    'discrete2': np.repeat(["a", "b"], [580, 420]),
    'discrete3': np.repeat([6, 7], [100, 900])
})

# index of columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

# 1. Model the data
# Step 1: Prepare your data
# data = load_demo()
#
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]

# Step 2: Fit TVAE to your data
tvae = TVAESynthesizer()
print('Training tvae is starting')
tvae.fit(data, discrete_columns, model_summary=True, trans="VGM")
# tvae.fit(data, discrete_columns, epochs=5)
print('Training tvae is completed')

# 2. Generate synthetic data
samples_1 = tvae.sample(10)
#4. Save and load the synthesizer
samples_1.to_csv(tvae.logger.dirpath + "/" + "tvae_samples_" + tvae.logger.PID + "_" + tvae.logger.dt.now().strftime(tvae.logger.datetimeformat) + ".csv", index=False, header=True)

# # To save a trained tvae synthesizer
tvae.save(tvae.logger.dirpath + "/" + "tvae_model_" + tvae.logger.PID + "_" + tvae.logger.dt.now().strftime(tvae.logger.datetimeformat)+ ".pkl")
