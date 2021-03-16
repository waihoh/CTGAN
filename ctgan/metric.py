import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F ## For JSD
from sklearn.utils import resample

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

# KL-divergence formula
def kl_divergence(p, q):
    # TODO: how to handle q == 0?
    # set a small number for numerical stability.
    p[p < 1e-12] = 1e-12
    q[q < 1e-12] = 1e-12
    a = np.log(p)
    b = np.log(q)
    return np.sum(p * (a - b))
    # return np.sum(np.where(p != 0, p * np.log(p / q), 0))

###For continuous variables:
def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True) #u: sorted x; c: count
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n ## Pe(x) in paper; When x=xi, U(x-xi)=0.5
    def interpolate_(x_): #Pc(x)
        yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate_


def cumulative_continuous_kl(x,y,fraction=0.5):
    dx = np.diff(np.sort(np.unique(x))) #Delta_x = xi-x_{i-1}
    dy = np.diff(np.sort(np.unique(y)))
    ex = np.min(dx) #min_i{xi-x_{i-1}}
    ey = np.min(dy)
    e = np.min([ex,ey])*fraction # e should be smaller than ex and ey; so here multiply 0.5
    n = len(x)
    P = ecdf(x)
    Q = ecdf(y)
    p = P(x) - P(x-e)
    q = Q(x) - Q(x-e)
    p[p < 1e-12] = 1e-12
    q[q < 1e-12] = 1e-12
    KL = abs((1./n)*np.sum(np.log(p/q))-1) #eq.5 in paper
    return KL


# def continuous_probs(column,bins):
#     column = pd.Series(np.digitize(column, bins))
#     counts = column.value_counts()
#     freqs = {counts.index[i]: counts.values[i] for i in range(len(counts.index))}
#     for i in range(1, len(bins)+1):
#         if i not in freqs.keys():
#             freqs[i] = 0
#     sorted_freqs = {}
#     for k in sorted(freqs.keys()):
#         sorted_freqs[k] = freqs[k]
#     probs = []
#     for k,v in sorted_freqs.items():
#         probs.append(v/len(column))
#     return np.array(probs)


def KLD(real, fake, discrete_columns):
    KLD = []
    for column in fake.columns:
        column_fake = fake[column].values
        column_real = real[column].values
        if column in discrete_columns:
            # find list of all unique values
            column_real = column_real[~np.isnan(column_real)]
            column_fake = column_fake[~np.isnan(column_fake)]
            unique_list = []
            arrs = [np.unique(column_fake), np.unique(column_real)]
            for arr in arrs:
                for val in arr:
                    if val not in unique_list:
                        unique_list.append(val)
            # find probabilities of values according to order in unique_list
            fake_prob = discrete_probs(column_fake, unique_list)
            real_prob = discrete_probs(column_real, unique_list)
            KLD.append((kl_divergence(fake_prob, real_prob)+kl_divergence(real_prob,fake_prob)/2)
)
        else:
            # check whether indicator columns exist
            if column + '_cat' in fake.columns:
                column_fake = column_fake[fake[column + '_cat'] == 0]
                column_real = column_real[real[column + '_cat'] == 0]
                ## list all continuous variables for which 0 is meaningful
                column_list = ['b10','b11','b12number_1','b12number_2','b12number_3_5',
                               'b12number_4','b12number_6','b12number_7_8','c1c','c4c_1','b12b_1','b12b_2']
                if column in column_list:
                    column_fake = column_fake[column_fake >= 0]
                    column_real = column_real[column_real >= 0]
                else:
                    column_fake = column_fake[column_fake > 0]
                    column_real = column_real[column_real > 0]
                if len(column_fake) >= 1000 and len(column_real) >= 1000:
                    KLD.append((cumulative_continuous_kl(column_fake, column_real)+cumulative_continuous_kl(column_real, column_fake))/2)
                else:
                    KLD.append(np.nan)
            else:
                KLD.append((cumulative_continuous_kl(column_fake, column_real)+cumulative_continuous_kl(column_real, column_fake))/2)
            # maxval = max(max(column_real), max(column_fake))
            # minval = min(min(column_real), min(column_fake))
            # bins = np.linspace(start=minval, stop=maxval, num=20) ##Is number of bins too small?
            # bins = np.histogram_bin_edges(np.arange(minval, maxval), bins='auto')
            # fake_prob = continuous_probs(column_fake, bins)
            # real_prob = continuous_probs(column_real, bins)

            # mean_prob = (fake_prob+real_prob)/2

    # return np.mean(KLD), np.mean(JSD)
    return np.array(KLD)

def determine_threshold(data,n,discrete_columns,n_rep=1000):
    boot_KLD = np.zeros((n_rep,data.shape[1]))
    for i in np.arange(n_rep):
        boot1 = resample(data, replace=False, n_samples=n)
        boot2 = resample(data, replace=False, n_samples=n)
        boot_KLD[i]= KLD(boot1,boot2,discrete_columns)
   # return np.nanmean(boot_KLD,axis=0)
    return np.nanmean(boot_KLD,axis=0)+2*np.nanstd(boot_KLD,axis=0)
