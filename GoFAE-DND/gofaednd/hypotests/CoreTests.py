#!/usr/bin/env python

from hypotests.HTest import HTest
import torch
import numpy as np
from scipy.stats import norm
import math
from scipy import stats


class AndersonDarling(HTest):

    def __init__(self, emp_dist, n, device, new_stat, use_emp=False):
        self.optdir = -1.
        self.emp_dist = emp_dist
        self.n = n
        self.use_emp = use_emp
        self.device = device
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            coef = 1 + 0.75 / self.n + 2.25 / self.n ** 2
            AA = coef * tstat
            if AA < 0.2:
                pval = 1. - np.exp(-13.436 + 101.14 * AA - 223.73 * AA ** 2)
            elif AA < 0.34:
                pval = 1 - np.exp(-8.318 + 42.796 * AA - 59.938 * AA ** 2)
            elif AA < 0.6:
                pval = np.exp(0.9177 - 4.279 * AA - 1.38 * AA ** 2)
            elif AA < 10:
                pval = np.exp(1.2937 - 5.709 * AA + 0.0186 * AA ** 2)
            else:
                pval = 0
        return pval

    def teststat(self, x):
        n = x.shape[0]
        seq = [(2 * i - 1) for i in range(1, n + 1)]
        seq = torch.tensor(np.array(seq), device=self.device).view(-1, 1)
        x_sort = torch.sort(x, dim=0, descending=False)
        x_sort = x_sort[0].view(-1, 1)
        # Center the groups
        mean = x_sort.mean(0, keepdim=True)
        x_mean = x_sort - mean
        dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        p1 = dist.cdf(x_mean / torch.std(x_sort))
        p2 = dist.cdf(-x_mean / torch.std(x_sort))
        rTensor = torch.flip(p2, dims=(0, 1))
        h = seq * ((torch.log(p1 + 1e-8) + torch.log(rTensor + 1e-8)))
        A = -n - torch.mean(h)
        return A

    def is_maximization(self):
        return False

class CramerVonMises(HTest):

    def __init__(self, emp_dist, n, device, new_stat, use_emp=False):
        self.optdir = -1.
        self.emp_dist = emp_dist
        self.n = n
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):
        # want to minimize CVM
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            coef2 = 1 + 0.5 / self.n
            WW = coef2 * tstat
            if WW < 0.0275:
                pval = 1. - np.exp(-13.953 + 775.5 * WW - 12542.61 * WW ** 2)
            elif WW < 0.051:
                pval = 1 - np.exp(-5.903 + 179.546 * WW - 1515.29 * WW ** 2)
            elif WW < 0.092:
                pval = np.exp(0.886 - 31.62 * WW + 10.897 * WW ** 2)
            elif WW < 1.1:
                pval = np.exp(1.111 - 34.242 * WW + 12.832 * WW ** 2)
            else:
                pval = 7.37e-10
        return pval


    def teststat(self, x):
        n = x.shape[0]
        coef1 = torch.tensor(1 / (12 * n), dtype=torch.float, device=self.device)
        seq = [(2 * i - 1) / (2 * n) for i in range(1, n + 1)]
        seq = torch.tensor(np.array(seq), dtype=torch.float, device=self.device).view(-1, 1)
        x_sort = torch.sort(x, dim=0, descending=False)
        x_sort = x_sort[0].view(-1, 1)
        mean = x_sort.mean(0, keepdim=True)
        x_mean = x_sort - mean
        dist = torch.distributions.normal.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        p = dist.cdf(x_mean / torch.std(x_sort))
        self.W = coef1 + (p - seq).pow(2).sum()
        return self.W

    def is_maximization(self):
        return False

class EppsPulley1(HTest):

    def __init__(self, emp_dist, n, device, new_stat, use_emp=False):
        self.optdir = 1.
        self.emp_dist = emp_dist
        self.n = n
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = stats.percentileofscore(self.emp_dist, tstat) / 100.
        return pval

    def teststat(self, x):
        alpha=torch.tensor(1.0, device=self.device)
        n = x.shape[0]
        x_mean= x.mean(0, keepdim=True)
        s2 = ((x - x_mean)**2).sum()/n
        alpha2 = alpha**2
        xmat = (x.unsqueeze(1) - x)**2/(2.0*alpha2*s2)
        term1 = torch.exp(-xmat).sum()/(n**2)
        term2 = (x-x_mean)**2/(2.0*s2*(1.0+alpha2))
        term2 = torch.exp(-term2).sum()
        term2 = (term2*2.0)/(n * torch.sqrt(1.0 + 1.0/alpha2))
        term3 = 1.0/torch.sqrt(1.0 + 2.0/alpha2)
        TEP_og = term1 - term2 + term3
        self.TEP = -torch.log(n * TEP_og + 1e-8) # ensure log doesn't screw up
        return self.TEP

    def is_maximization(self):
        return True


class KolmogorovSmirnov(HTest):

    def __init__(self, emp_dist, n, device, new_stat, use_emp=False):
        self.optdir = -1.
        self.emp_dist = emp_dist
        self.n = n
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):

        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            if (self.n <= 100):
                Kd = tstat
                nd = self.n
            else:
                Kd = tstat * ((self.n / 100) ** (.49))
                nd = 100

            pval = np.exp(-7.01256 * Kd ** 2 * (nd + 2.78019) + 2.99587 * Kd * math.sqrt(
                nd + 2.78019) - 0.122119 + 0.974598 / math.sqrt(nd) + 1.67997 / nd)
            if pval > 0.1:
                KK = (math.sqrt(self.n) - 0.01 + 0.85 / math.sqrt(self.n)) * tstat
                if KK <= 0.302:
                    pval = 1
                elif KK <= 0.5:
                    pval = 2.76773 - 19.828315 * KK + 80.709644 * KK ** 2 - 138.55152 * KK ** 3 + 81.218052 * KK ** 4
                elif KK <= 0.9:
                    pval = -4.901232 + 40.662806 * KK - 97.490286 * KK ** 2 + 94.029866 * KK ** 3 - 32.355711 * KK ** 4
                elif KK <= 1.31:
                    pval = 6.198765 - 19.558097 * KK + 23.186922 * KK ** 2 - 12.234627 * KK ** 3 + 2.423045 * KK ** 4
                else:
                    pval = 0
        return pval

    def teststat(self, x):
        n = x.shape[0]
        x_sort = torch.sort(x, dim=0, descending=False)
        x_sort = x_sort[0].view(-1, 1)
        mean = x_sort.mean(0, keepdim=True)
        x_mean = x_sort - mean
        dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        p = dist.cdf(x_mean / torch.std(x_sort))
        seq_plus = [i / n for i in range(1, n + 1)]
        seq_minus = [(i - 1) / n for i in range(1, n + 1)]
        seq_plus = torch.tensor(np.array(seq_plus), dtype=torch.float, device=self.device).view(-1, 1)
        seq_minus = torch.tensor(np.array(seq_minus), dtype=torch.float, device=self.device).view(-1, 1)
        Dplus = torch.max(seq_plus - p)
        Dminus = torch.max(p - seq_minus)
        self.K = torch.max(Dplus, Dminus)
        return self.K

    def is_maximization(self):
        return False


class ShapiroFrancia(HTest):

    def __init__(self, emp_dist, sf_wts, new_mu, new_sigma, device, new_stat, use_emp=False):
        self.optdir = 1.
        self.emp_dist = emp_dist
        self.sf_wts = sf_wts
        self.sf_mu = new_mu
        self.sf_sigma = new_sigma
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):
        # want to maximize SF
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            newSFstat = np.log(1. - tstat)
            normalSFstat = (newSFstat - self.sf_mu) / self.sf_sigma
            pval = 1.0 - stats.norm.cdf(normalSFstat, loc=0., scale=1.)
        return pval

    def teststat(self, x):
        X_sort = torch.sort(x, dim=0, descending=False)
        X_sort = X_sort[0].view(-1, 1)
        avg = torch.mean(X_sort)
        a = torch.sum(torch.mul(X_sort, self.sf_wts))
        b = torch.sum((X_sort - avg) ** 2)
        self.W = torch.div(a ** 2, b)
        return self.W

    def is_maximization(self):
        return True


class ShapiroWilk(HTest):

    def __init__(self, emp_dist, sw_wts, new_mu, new_sigma, device, new_stat, use_emp=False):
        self.optdir = 1.
        self.emp_dist = emp_dist
        self.sw_wts = sw_wts
        self.new_mu = new_mu
        self.new_sigma = new_sigma
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return True

    def pval(self, inputs):
        # want to maximize SW
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            newSWstat = np.log(1. - tstat)
            normalSWstat = (newSWstat - self.new_mu) / self.new_sigma
            pval = 1.0 - stats.norm.cdf(normalSWstat, loc=0., scale=1.)
        return pval

    def teststat(self, x):
        X_sort = torch.sort(x, dim=0, descending=False)
        X_sort = X_sort[0].view(-1, 1)
        avg = torch.mean(X_sort)
        a = torch.sum(torch.mul(X_sort, self.sw_wts))
        b = torch.sum((X_sort - avg) ** 2)
        self.W = torch.div(a ** 2, b)
        return self.W

    def is_maximization(self):
        return True



class HenzeZirkler(HTest):

    def __init__(self, emp_dist, device, new_stat, n, m, use_emp):
        self.optdir = -1.
        self.emp_dist = emp_dist
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

        n1 = n
        p1 = m
        b = (1.0 / np.sqrt(2.0)) * (((2.0 * p1 + 1.0) / 4.0) ** (1.0 / (p1 + 4.0))) * (n1 ** (1.0 / (p1 + 4.0)))
        wb = (1.0 + b ** 2.0) * (1.0 + 3.0 * b ** 2.0)
        a = 1.0 + 2.0 * b ** 2.0
        mu = 1.0 - a ** (-p1 / 2.0) * (
                    1.0 + p1 * (b ** 2.0) / a + (p1 * (p1 + 2.0) * (b ** 4.0)) / (2.0 * a ** 2.0))  # HZ mean
        si2 = 2.0 * (1.0 + 4.0 * b ** 2.0) ** (-p1 / 2.0) + 2.0 * a ** (-p1) * (
                    1.0 + (2.0 * p1 * b ** 4.0) / (a ** 2.0) + (3.0 * p1 * (p1 + 2.0) * b ** 8.0) / (4.0 * a ** 4.0)) - \
              4.0 * (wb ** (-p1 / 2.0)) * (1.0 + (3.0 * p1 * b ** 4.0) / (2.0 * wb) + (p1 * (p1 + 2.0) * b ** 8.0) / (
                    2.0 * wb ** 2.0))  # HZ Variance
        pmu = np.log(np.sqrt((mu ** 4.0) / (si2 + mu ** 2.0)))  # lognormal HZ mean
        psi = np.sqrt(np.log((si2 + mu ** 2.0) / mu ** 2.0))  # lognormal HZ variance
        b = torch.tensor(b, device=self.device)
        self.b = b
        self.wb = wb
        self.a = a
        self.pmu = pmu
        self.psi = psi

    def is_univariate(self):
        return False

    def pval(self, inputs):
        # want to minimize HZ
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            shape = self.psi
            scale = np.exp(self.pmu)
            pval = 1.0 - stats.lognorm.cdf(tstat, shape, 0.0, scale)
        return pval

    def teststat(self, x):
        n, m = x.shape
        f1 = torch.tensor(np.repeat(1.0, n).reshape(1, n), dtype=torch.float, device=self.device)
        # Center the groups
        mean = x.mean(0, keepdim=True)
        x_mean = x - mean
        # Calculate the Sigma of groups (maybe move self.eps into Lambda
        sigma = x_mean.t().matmul(x_mean) / x.size(0)
        # Sigma inverse
        sigmainv = torch.inverse(sigma)
        Di = x_mean.matmul(sigmainv).matmul(x_mean.t()).diag()
        Y = x.matmul(sigmainv).matmul(x.t())
        Djk = (-2 * Y.t()) + Y.t().diag().view(-1, 1).matmul(f1) + f1.t().matmul(Y.t().diag().reshape(1, -1))
        self.HZ = n * (1 / (n ** 2) * torch.exp(- (self.b ** 2) / 2 * Djk).sum() - 2 * \
                  ((1 + (self.b ** 2)).pow(- m / 2)) * (1 / n) * torch.exp(- ((self.b ** 2) / (2 * \
                                                                                     (1 + (self.b ** 2)))) * Di).sum() + (
                      (1 + (2 * (self.b ** 2))).pow(- m / 2)))
        return self.HZ

    def is_maximization(self):
        return False


class MardiaSkew(HTest):

    def __init__(self, emp_dist, n_z, device, new_stat, use_emp):
        self.optdir = -1.
        self.emp_dist = emp_dist
        self.n_z = n_z
        self.device = device
        self.use_emp = use_emp
        self.new_stat=new_stat

    def is_univariate(self):
        return False

    def pval(self, inputs):
        # want to minimize mardia_skew
        tstat = inputs.detach().cpu().numpy()
        if use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            df = self.n_z * (self.n_z + 1) * (self.n_z + 2) / 6
            pval = 1. - stats.chi2.cdf(tstat, df)
        return pval

    def teststat(self, x):
        n, m = x.shape
        # Center the groups
        mean = x.mean(0, keepdim=True)
        x_mean = x - mean
        # Calculate the Sigma of groups (maybe move self.eps into Lambda
        sigma = x_mean.t().matmul(x_mean) / x.size(0)
        # Sigma inverse
        sigmainv = torch.inverse(sigma)
        D = x_mean.matmul(sigmainv).matmul(x_mean.t())
        g1p = torch.sum(D.pow(3)) / n ** 2
        g2p = D.pow(2).diag().sum() / n
        # skew
        # don't want kurtosis as test statistic is normally distributed
        self.skew = (n * g1p) / 6.
        return self.skew

    def is_maximization(self):
        return False


class Royston(HTest):

    def __init__(self, emp_dist, e, device, sf, sw, new_stat, n, use_emp):
        self.emp_dist = emp_dist
        self.optdir = -1.
        self.e = e
        self.device = device
        self.sf = sf
        self.sw = sw
        self.use_emp = use_emp
        self.new_stat=new_stat
        q = np.log(n)
        g = 0.0
        m = -1.5861 + (-0.31082 * q) + (-0.083751 * q ** 2) + (0.0038915 * q ** 3)
        s = np.exp(-0.4803 + (-0.082676 * q) + (0.0030302 * q ** 2))
        u = 0.715
        v = 0.21364 + (0.015124 * q ** 2) + (-0.0018034 * q ** 3)
        self.g = g
        self.m = m
        self.s = s
        self.u = u
        self.v = v

    def is_univariate(self):
        return False

    def pval(self, inputs):
        # want to minimize Royston
        tstat = inputs.detach().cpu().numpy()
        if self.use_emp:
            pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        else:
            df = self.e.detach().cpu().numpy()
            pval = 1. - stats.chi2.cdf(tstat, float(df))
        return pval

    def teststat(self, x):
        n = x.shape[0]
        p = x.shape[1]
        q = np.log(n)
        z = []
        for i in range(p):
            K = kurtosis(x[:, i])
            if K > 3.0:
                z.append((((torch.log(1 - self.sf.teststat(x[:, i])) + self.g) - self.m) / self.s).view(-1, 1))
            else:
                z.append((((torch.log(1 - self.sw.teststat(x[:, i])) + self.g) - self.m) / self.s).view(-1, 1))

        output_tensor = torch.cat(z, dim=1)
        c = corr(x)
        c = c.fill_diagonal_(1., wrap=False)
        NC = (c ** 5) * (1 - (self.u * (1 - c) ** self.u) / self.v)
        T = NC.sum() - p
        mC = T / (p ** 2 - p)
        edf = p / (1 + (p - 1) * mC)
        dist = torch.distributions.normal.Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        Res = dist.icdf(dist.cdf(-output_tensor.double()) / 2) ** 2
        H = (edf * Res.sum()) / p
        self.H=H
        self.edf=edf

        return self.H, self.edf

    def is_maximization(self):
        return False

    def kurtosis(x):
        x = x.detach().cpu().numpy()
        n = x.shape[0]
        k = n * np.sum((x - np.mean(x)) ** 4) / (np.sum((x - np.mean(x)) ** 2) ** 2)
        return k

    def cov(x):
        mean = x.mean(0, keepdim=True)
        x_mean = x - mean
        return x_mean.t().matmul(x_mean) / (x.size(0) - 1)

    def corr(x):
        sig = cov(x)
        sinv = sig.diag().rsqrt().diag()
        return sinv.matmul(sig.matmul(sinv))

def SWF_weights(N, device):
    # N = batch size

    p = range(1,(N+1),1)
    mtilde = norm.ppf((p - np.repeat((3.0/8.0),N))/(N + (1.0/4.0)))
    mtilde = np.reshape(mtilde,[N,1])

    weights = np.zeros([N,1])

    # Shapiro-Wilk
    c = 1.0/np.sqrt(np.dot(np.transpose(mtilde), mtilde)) * mtilde
    u = 1.0/np.sqrt(N)

    PolyCoef_1 = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, c[-1].item()]
    PolyCoef_2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, c[-2].item()]
    PolyCoef_3 = [-0.0006714, 0.0250540, -0.39978, 0.54400]
    PolyCoef_4 = [-0.0020322, 0.0627670, -0.77857, 1.38220]
    PolyCoef_5 = [0.00389150, -0.083751, -0.31082, -1.5861]
    PolyCoef_6 = [0.00303020, -0.082676, -0.48030]
    PolyCoef_7 = [0.459, -2.273]

    weights[-1] = np.polyval(PolyCoef_1, u)
    weights[0] = -weights[-1]

    if N > 5:
        weights[-2] = np.polyval(PolyCoef_2, u)
        weights[1] = -weights[-2]

        count = 2
        phi = (np.dot(np.transpose(mtilde), mtilde)-2.0*mtilde[-1]**2 - 2.0*mtilde[-2]**2)/(1.0 - 2.0*weights[-1]**2 - 2.0*weights[-2]**2)

    weights[count:-2] = mtilde[count:-2]/np.sqrt(phi)

    # for normalized W
    newn = np.log(N)
    sw_mu = np.polyval(PolyCoef_5, newn)
    sw_sigma = np.exp(np.polyval(PolyCoef_6, newn))

    # Shaprio-Francia
    nu = newn
    u1 = np.log(nu) - nu
    u2 = np.log(nu) + 2/nu
    sf_mu = -1.2725 + (1.0521 * u1)
    sf_sigma = 1.0308 - (0.26758 * u2)

    return torch.tensor(weights, dtype=torch.float, device=device) , sw_mu, sw_sigma, torch.tensor(c, dtype=torch.float, device=device), sf_mu, sf_sigma
