#!/usr/bin/env python

from hypotests.HTest import HTest
import sys
import torch
import numpy as np
import pandas as pd
import math
from scipy.stats import chisquare
from scipy import stats
import os


class ChiSquareUnif(HTest):

    def __init__(self, method="sturges"):
        self.chisq_pval = None
        self.chisq_tstat = None
        self.method = method

    def is_univariate(self):
        return True

    def pval(self, inputs):
        raise NotImplementedError("Call teststat method which computes both chisq_tstat and chisq_pval.")

    def teststat(self, x):

        n = x.shape[0]
        # Calculate bins
        if self.method == 'sturges':
            num_bins = math.ceil(math.log2(n)) + 1
        elif self.method == 'simple':
            if n > 100:
                num_bins = math.floor(10 * math.log10(n))
            else:
                num_bins = math.floor(2 / math.sqrt(n))
        else:
            sys.exit('Method not implemented')

        bin_counts, _ = np.histogram(x, num_bins)
        chisq_tstat, chisq_pval = chisquare(bin_counts)
        self.chisq_tstat = chisq_tstat
        self.chisq_pval = chisq_pval
        return self.chisq_tstat

    def is_maximization(self):
        return False


class KolmogorovSmirnovUnif(HTest):

    def __init__(self, device, emp_dist):
        self.optdir = -1.
        self.device = device
        self.emp_dist = emp_dist

    def is_univariate(self):
        return True

    def pval(self, inputs):
        tstat = inputs.detach().cpu().numpy()
        pval = 1. - stats.percentileofscore(self.emp_dist, tstat) / 100.
        return pval

    def teststat(self, x):
        n = x.shape[0]
        x_sort = torch.sort(x, dim=0, descending=False)
        x_sort = x_sort[0].view(-1, 1)
        x_sort_min = torch.min(x_sort)
        x_sort_max = torch.max(x_sort)
        dist = torch.distributions.uniform.Uniform(torch.tensor([0.0], device=self.device), torch.tensor([1.0],
                                                                                                         device=self.device))  # testing U(0,1) NOT U(min(x), max(x))
        p = dist.cdf(x_sort)
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

    @staticmethod
    def get_ks_unif_info(path, samp_size):
        emp_dist_path1 = os.path.join(path, 'ks_unif', str('sample_size_') + str(samp_size), 'emp_dist_tstat.csv')
        emp_dist = np.array(pd.read_csv(emp_dist_path1))
        return emp_dist
