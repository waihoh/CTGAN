import torch
import torch.nn.functional as F ## For JSD
### calculate Jensen--Shannon Divergence
def js_div(p_output,q_output,get_softmax=True):
    criterion = torch.nn.KLDivLoss(reduction='batchmean') #the sum of the output will be divided by batchsize
    if get_softmax:
        p_output = F.softmax(p_output,dim=1) ## transform to probability
        q_output = F.softmax(q_output,dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return(criterion(log_mean_output,p_output)+criterion(log_mean_output,q_output))/2

def kl_div(p_output,q_output,get_softmax=True):
    criterion = torch.nn.KLDivLoss(reduction='batchmean')  # the sum of the output will be divided by batchsize
    if get_softmax:
        p_output = F.softmax(p_output,dim=1) ## transform to probability
        q_output = F.softmax(q_output,dim=1)
    return criterion(p_output.log(),q_output)


# For discrete data
def discret_probs(column):
    counts = column.value_counts()
    freqs = {counts.index[i]: counts.values[i] for i in range(len(counts.index))}
    probs = []
    for k, v in freqs.items():
        probs.append(v / len(column))
    return np.array(probs)


# For continuous data
bins = np.arange(-50, 13800, 10)

real_inds = pd.DataFrame(np.digitize(real_data['NUMLABEVENTS'], bins), columns=['inds'])
syn_inds = pd.DataFrame(np.digitize(syn_data['NUMLABEVENTS'], bins), columns=['inds'])


def identify_probs(table, column):
    counts = table[column].value_counts()
    freqs = {counts.index[i]: counts.values[i] for i in range(len(counts.index))}
    for i in range(1, len(bins) + 1):
        if i not in freqs.keys():
            freqs[i] = 0
    sorted_freqs = {}
    for k in sorted(freqs.keys()):
        sorted_freqs[k] = freqs[k]
    probs = []
    for k, v in sorted_freqs.items():
        probs.append(v / len(table[column]))
    return sorted_freqs, np.array(probs)


real_p = identify_probs(real_inds, 'inds')[1]
syn_p = identify_probs(syn_inds, 'inds')[1]

# KL-divergence formula
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
