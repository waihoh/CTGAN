import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F ## For JSD
### calculate Jensen--Shannon Divergence
## softmax is not valid
# def js_div(p_output,q_output,get_softmax=True):
#     criterion = torch.nn.KLDivLoss(reduction='batchmean') #the sum of the output will be divided by batchsize
#     if get_softmax:
#         p_output = F.softmax(p_output,dim=1) ## transform to probability
#         q_output = F.softmax(q_output,dim=1)
#     log_mean_output = ((p_output + q_output) / 2).log()
#     return(criterion(log_mean_output,p_output)+criterion(log_mean_output,q_output))/2
#
# def kl_div(p_output,q_output,get_softmax=True):
#     criterion = torch.nn.KLDivLoss(reduction='batchmean')  # the sum of the output will be divided by batchsize
#     if get_softmax:
#         p_output = F.softmax(p_output,dim=1) ## transform to probability
#         q_output = F.softmax(q_output,dim=1)
#     return criterion(p_output.log(),q_output)

# For discrete data
def discrete_probs(column):
    column = pd.Series(column)
    counts = column.value_counts()
    freqs ={counts.index[i]: counts.values[i] for i in range(len(counts.index))}
    probs = []
    for k,v in freqs.items():
        probs.append(v/len(column))
    return np.array(probs)

def continuous_probs(column,bins):
    column = pd.Series(np.digitize(column, bins))
    counts = column.value_counts()
    freqs = {counts.index[i]: counts.values[i] for i in range(len(counts.index))}
    for i in range(1, len(bins)+1):
        if i not in freqs.keys():
            freqs[i] = 0
    sorted_freqs = {}
    for k in sorted(freqs.keys()):
        sorted_freqs[k] = freqs[k]
    probs = []
    for k,v in sorted_freqs.items():
        probs.append(v/len(column))
    return np.array(probs)

# KL-divergence formula
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def KLD_JSD(fake,real,discrete_columns):
    KLD = []
    JSD = []
    for column in fake.columns:
        column_fake = fake[column].values
        column_real = real[column].values
        bins = np.arange(min(min(column_real),min(column_fake)),max(max(column_real),max(column_fake)), 20)
        if column in discrete_columns:
            fake_prob = discrete_probs(column_fake)
            real_prob = discrete_probs(column_real)
        else:
            fake_prob = continuous_probs(column_fake,bins)
            real_prob = continuous_probs(column_real,bins)
        mean_prob = (fake_prob+real_prob)/2
        JSD.append((kl_divergence(fake_prob,mean_prob)+kl_divergence(real_prob,mean_prob))/2)
        KLD.append(kl_divergence(fake_prob,real_prob))


    return sum(KLD),sum(JSD)
