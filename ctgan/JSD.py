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
