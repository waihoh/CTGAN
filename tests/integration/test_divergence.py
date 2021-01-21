import pandas as pd
import numpy as np
import ctgan.metric as M

np.random.seed(0)

# index of columns
discrete_columns = ['discrete1']

# reference data
data_continuous_a = np.random.uniform(low=0, high=1, size=1000)
data_discrete_a = np.repeat([1,2], [950, 50])
np.random.shuffle(data_discrete_a)
list_discrete_a = data_discrete_a.tolist()

# ------------------------------------------------------------------------------------------------
# using a toy example to test tablegan
data_a = pd.DataFrame({
    'continuous1': data_continuous_a,
    'discrete1': data_discrete_a
})

# ------------------------------------------------------------------------------------------------
# data_b is an exact copy of data_a.
# We anticipate the KL or JS divergence to be minimum
data_b = data_a.copy()

KL_loss_ab, JS_loss_ab = M.KLD_JSD(data_a, data_b, discrete_columns)
print("Case 1: Minimum divergence")
print(KL_loss_ab, JS_loss_ab)

# ------------------------------------------------------------------------------------------------
# data_c will be non-overlapping with data_a2.
# We anticipate the KL or JS divergence to be maximum
data_list_a2 = [4] * 1000
data_discrete_a2 = np.array(data_list_a2)
data_a2 = pd.DataFrame({
    'continuous1': data_continuous_a,
    'discrete1': data_discrete_a2
})

data_continuous_c = np.random.uniform(low=3, high=4, size=1000)
data_list_c = [3] * 1000
data_discrete_c = np.array(data_list_c)
data_c = pd.DataFrame({
    'continuous1': data_continuous_c,
    'discrete1': data_discrete_c
})

KL_loss_a2c, JS_loss_a2c = M.KLD_JSD(data_a2, data_c, discrete_columns)
print("Case 2: Maximum divergence")
print(KL_loss_a2c, JS_loss_a2c)


# ------------------------------------------------------------------------------------------------
# data_d will be overlapping with data_a
# We anticipate the KL or JS divergence to be between the minimum and maximum values.
data_continuous_d = np.random.uniform(low=0.5, high=1.5, size=1000)
list_discrete_d = []
rand_num_list = np.random.uniform(size=1000).tolist()
for i in range(len(list_discrete_a)):
    val = list_discrete_a[i]
    # flip the value half of the time
    if rand_num_list[i] < 0.5:
        if val == 1:
            val = 2
        else:
            val = 1
    list_discrete_d.append(val)
data_discrete_d = np.array(list_discrete_d)

data_d = pd.DataFrame({
    'continuous1': data_continuous_d,
    'discrete1': data_discrete_d
})

print("Case 3: Some overlap")
KL_loss_ad, JS_loss_ad = M.KLD_JSD(data_a, data_d, discrete_columns)
print(KL_loss_ad, JS_loss_ad)
