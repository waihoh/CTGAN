import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F ## For JSD

## calculate Jensen--Shannon Divergence
# softmax is not valid
# def js_div(p_output,q_output,get_softmax=True):
#     # p_output = torch.transpose(p_output, 0, 1)
#     # q_output = torch.transpose(q_output, 0, 1)
#     criterion = torch.nn.KLDivLoss(reduction='batchmean') #the sum of the output will be divided by batchsize
#     if get_softmax:
#         p_output = F.softmax(p_output, dim=1) ## transform to probability
#         q_output = F.softmax(q_output, dim=1)
#     log_mean_output = ((p_output + q_output) / 2).log()
#     return(criterion(log_mean_output, p_output)+criterion(log_mean_output, q_output))/2
#
# def kl_div(p_output,q_output,get_softmax=True):
#     # p_output = torch.transpose(p_output, 0, 1)
#     # q_output = torch.transpose(q_output, 0, 1)
#     criterion = torch.nn.KLDivLoss(reduction='batchmean')  # the sum of the output will be divided by batchsize
#     if get_softmax:
#         p_output = F.softmax(p_output, dim=1) ## transform to probability
#         q_output = F.softmax(q_output, dim=1)
#     return criterion(p_output.log(), q_output)


# For discrete data
def discrete_probs(column, unique_list):
    # find probability in the order of unique_list
    column = pd.Series(column)
    counts = column.value_counts()
    # freqs = {counts.index[i]: counts.values[i] for i in range(len(counts.index))}
    # probs = []
    # for k, v in freqs.items():
    #     probs.append(v/len(column))

    probs = []
    total_length = len(column)
    for i in unique_list:
        val = 0
        if i in counts.index:
            val = counts[i]
        probs.append(val/total_length)

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
    # TODO: how to handle q == 0?
    # set a small number for numerical stability.
    p[p < 1e-8] = 1e-8
    q[q < 1e-8] = 1e-8
    a = np.log(p)
    b = np.log(q)
    return np.sum(p * (a - b))
    # return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def KLD_JSD(fake, real, discrete_columns):
    KLD = []
    JSD = []
    for column in fake.columns:
        column_fake = fake[column].values
        column_real = real[column].values
        if column in discrete_columns:
            # find list of all unique values
            unique_list = []
            arrs = [np.unique(column_fake), np.unique(column_real)]
            for arr in arrs:
                for val in arr:
                    if val not in unique_list:
                        unique_list.append(val)
            # find probabilities of values according to order in unique_list
            fake_prob = discrete_probs(column_fake, unique_list)
            real_prob = discrete_probs(column_real, unique_list)
        else:
            maxval = max(max(column_real), max(column_fake))
            minval = min(min(column_real), min(column_fake))
            # bins = np.linspace(start=minval, stop=maxval, num=20) ##Is number of bins too small?
            bins = np.histogram_bin_edges(np.arange(minval, maxval), bins='auto')
            print("number of bins:", len(bins))
            fake_prob = continuous_probs(column_fake, bins)
            real_prob = continuous_probs(column_real, bins)
        mean_prob = (fake_prob+real_prob)/2

        JSD.append((kl_divergence(fake_prob, mean_prob)+kl_divergence(real_prob, mean_prob))/2)
        KLD.append(kl_divergence(fake_prob, real_prob))

    # return np.mean(KLD), np.mean(JSD)
    return sum(KLD), sum(JSD)

