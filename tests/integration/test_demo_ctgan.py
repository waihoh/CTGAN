# test the GitHub demo
import os
from ctgan import load_demo
from ctgan import CTGANSynthesizer
import pandas as pd
import numpy as np

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

# # 1. Model the data
# # Step 1: Prepare your data
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

print('IN DEMO')
print(discrete_columns)

# Step 2: Fit CTGAN to your data
ctgan = CTGANSynthesizer()

# Create a new folder to save the training results
ctgan.logger.change_dirpath(ctgan.logger.dirpath + "/CTGAN_" + ctgan.logger.PID) ## create a folder with PID

ctgan.fit(data, discrete_columns, model_summary=False, trans="VGM")

# print("before saving, does file exist?", os.path.exists(path_to_a_folder))

# 2. Generate synthetic data
samples_1 = ctgan.sample(10, condition_column='discrete1', condition_value=1)

# Saving
samples_1.to_csv(ctgan.logger.dirpath + "/" + "ctgan_samples_" + ctgan.logger.PID + "_" + ctgan.logger.dt.now().strftime(ctgan.logger.datetimeformat) + ".csv", index=False, header=True)

# To save a trained ctgan synthesizer
ctgan.save(ctgan.logger.dirpath + "/" + "ctgan_model_" + ctgan.logger.PID + "_" + ctgan.logger.dt.now().strftime(ctgan.logger.datetimeformat)+ ".pkl")

# # NOTE: We'll see warnings:
# # UserWarning: Couldn't retrieve source code for container of type ... . It won't be checked for correctness upon loading.
# #   "type " + obj.__name__ + ". It won't be checked "
#
