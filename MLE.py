import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats

class MLE():
    def __init__(self, data):
        self.states = data[self.data.columns[self.data.columns.str.startswith('state')]]
        self.target = data[["target"]]

    def mle_LinearRegression(self, coefficients):
        # coefficients[-2]: bias term, coefficients[-1]: standard deviation
        yhat = np.dot(self.states,coefficients[:-2]) + coefficients[-2]

        negative_log_likelihood = -np.sum(stats.norm.logpdf(self.target, loc=yhat, scale = coefficients[-1]))

        return negative_log_likelihood
    
    def optimize_mle(self):
        initial_guess = np.random.rand(len(self.states) + 2)
        results = minimize(fun = self.mle_LinearRegression, x0=initial_guess, method="Nelder-Mead")
        return results