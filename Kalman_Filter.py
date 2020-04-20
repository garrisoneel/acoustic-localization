import numpy as np
import pandas as pd

class Kalman():
    def __init__(self):
        self.A = None
        self.B = None
        self.P_prev = None
        self.P_0 = None
        self.P_priori = None
        self.K = None
        self.Q = None
        self.R = None
        self.H = None
        self.n = self.A.shape[0]

    # def gauss(self,x,mu,sigma):
    #     return (1./np.sqrt(2.*np.pi*np.square(sigma)))*np.exp(-0.5*np.square(x-mu)/(2.*np.square(sigma)))
    
    def time_update(self, states, inputs):
        self.state_prior = np.matmul(self.A,states) + np.matmul(self.B, inputs)

        if self.P_prev is None:
            self.P_priori = np.matmul(np.matmul(self.A,self.P_priori),self.A.transpose()) + self.Q
        else:
            self.P_priori = np.matmul(np.matmul(self.A,self.P_prev),self.A.transpose()) + self.Q
        
        return self.state_prior, self.P_priori
    
    def measurement_update(self, state_prior, P_priori, z):
        K1 = np.matmul(P_priori,self.H.transpose())
        k2 = np.linalg.inv(np.matmul(np.matmul(self.H,P_priori),self.H.transpose()) + self.R)
        self.K = np.matmul(k1,k2)

        measurement_error = np.matmul(self.K,  z - np.matmul(self.H,state_prior))

        state_estimation = state_prior + measurement_error

        p1 = np.identity(self.n) - np.matmul(self.K,self.H)

        self.P_prev = np.matmul(p1, P_priori)

        return state_estimation, self.P_prev