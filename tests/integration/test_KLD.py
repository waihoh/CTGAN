import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
from kneed import KneeLocator #
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

###For continuous variables:
def ecdf(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True) #u: sorted x; c: count
    n = len(x)
    y = (np.cumsum(c) - 0.5)/n ## Pe(x) in paper; When x=xi, U(x-xi)=0.5; From 0.5/n, to 1-(0.5/n)
    def interpolate_(x_): #Pc(x)
        yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
        return yinterp
    return interpolate_

def ecdf1(x):
    x = np.sort(x)
    u, c = np.unique(x, return_counts=True) #u: sorted x; c: count
    n = len(x)
    y = np.cumsum(c)/n ## Pe(x) in paper; When x=xi, U(x-xi)=0.5; From 0.5/n, to 1-(0.5/n)
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
    KL = abs((1./n)*np.sum(np.log(p/q))-1) #eq.5 in paper #KL will increase if we remove -0.5
    return KL

def testecdf(x,fraction=0.5):
    dx = np.diff(np.sort(np.unique(x)))  # Delta_x = xi-x_{i-1}
    ex = np.min(dx)  # min_i{xi-x_{i-1}}
    e = np.min(ex) * fraction  # e should be smaller than ex and ey; so here multiply 0.5
    P = ecdf(x)
    P1 = ecdf1(x)
    p = P(x) - P(x - e) ## only affect the probability of minimum value of x; remove -0.5, then from 0.5/n to 1/n
    p1 = P1(x) - P1(x-e)
    return [p,p1]

#print(testecdf(np.random.normal(size=10)))
# KL-divergence formula
def kl_divergence(p, q):
    # TODO: how to handle q == 0?
    # set a small number for numerical stability.
    p[p < 1e-12] = 1e-12
    q[q < 1e-12] = 1e-12
    a = np.log(p)
    b = np.log(q)
    return np.sum(p * (a - b))





###D(a||b)


def p(x):
    return norm.pdf(x, 0, 2)

def q(x):
    return norm.pdf(x, 2, 2)

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

KL_int, err = quad(KL, -10, 10) # integral from -10 to 10
print('KL: ', KL_int )

## sample size

x = np.arange(-10, 10, 0.001)
p = norm.pdf(x, 0, 2)
px = p/sum(p)
q = norm.pdf(x, 2, 2)
qx = q/sum(q)


print("KL is: ", kl_divergence(px,qx))

n = np.arange(10, 10000, 10)
KLhat = []
for i in np.arange(len(n)):
    a = np.random.normal(0,2,n[i])
    b = np.random.normal(2,2,n[i])
    KLhat.append(cumulative_continuous_kl(a,b,fraction=0.5))
plt.plot(n,KLhat)
plt.axhline(y=KL_int,color="r")
plt.show();

# n = np.arange(10, 1000, 10)
# KLhat1 = []
# KLhat5 = []
# KLhat9 = []
# for i in np.arange(len(n)):
#     a = np.random.normal(0,2,n[i])
#     b = np.random.normal(2,2,n[i])
#     KLhat1.append(cumulative_continuous_kl(a,b,fraction=0.1))
#     KLhat5.append(cumulative_continuous_kl(a, b, fraction=0.5))
#     KLhat9.append(cumulative_continuous_kl(a, b, fraction=0.9))
#
#
# # kl = KneeLocator(n, KLhat, curve="convex", direction="decreasing")
# # kl.elbow
# # print(kl.elbow)
#
# fig, ax = plt.subplots(1,3,figsize=(10,3))
# ax[0].plot(n,KLhat1)
# ax[0].axhline(y=KL_int,color="r")
# ax[0].set_title('fraction=0.1')
# ax[1].plot(n,KLhat5)
# ax[1].axhline(y=KL_int,color="r")
# ax[1].set_title('fraction=0.5')
# ax[2].plot(n,KLhat9)
# ax[2].axhline(y=KL_int,color="r")
# ax[2].set_title('fraction=0.9')
#
# plt.show();
# #
# #

