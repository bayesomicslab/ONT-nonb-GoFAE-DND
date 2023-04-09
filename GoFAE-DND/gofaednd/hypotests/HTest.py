#!/usr/bin/env python


class HTest(object):

    def is_univariate(self):
        raise NotImplementedError("Hypothesis tests must implement this method.")

    def pval(self, **kwargs):
        raise NotImplementedError("Hypothesis tests must implement this method.")

    def teststat(self, **kwargs):
        raise NotImplementedError("Hypothesis tests must implement this method.")

    def is_maximization(self):
        raise NotImplementedError("Hypothesis tests must implement this method.")