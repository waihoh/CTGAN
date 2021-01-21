import os
from ctgan import load_demo
from ctgan.tablegan import TableganSynthesizer
from ctgan.tablegan import determine_layers
from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator
import numpy as np
import pandas as pd

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

# # Step 2: Fit tableGan to your data
tablegan = TableganSynthesizer()
print('Training tablegan is starting')
# # NOTE: This runs much slower than ctgan and tvae
## use VGM transformation
tablegan.fit(data, discrete_columns=discrete_columns, model_summary=True,trans="VGM")
print('Training tablegan is completed')
#
# # 2. Generate synthetic data
#samples_1 = tablegan.sample(10, condition_column='discrete1', condition_value=1)
samples_1 = tablegan.sample(10, condition_column=None, condition_value=None)
#Save and load the synthesizer
samples_1.to_csv(tablegan.logger.dirpath + "/" + "tablegan_samples_" + tablegan.logger.PID + "_" + tablegan.logger.dt.now().strftime(tablegan.logger.datetimeformat) + ".csv", index=False, header=True)

# # To save a trained ctgan synthesizer
tablegan.save(tablegan.logger.dirpath + "/" + "tablegan_model_" + tablegan.logger.PID + "_" + tablegan.logger.dt.now().strftime(tablegan.logger.datetimeformat)+ ".pkl")

