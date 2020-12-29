# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:31:35 2020

@author: sybil
Implementing a Stochastic Adaptive Linear Neuron
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import seed

class AdalineSGD(object):
    """ADaptive LInear NEuron classifier.
    Parameters
    ---------------
    eta: float
        Learning rate between 0.0 and 1.0
    n_iter: int
            Passes over the training dataset
            
    Attributes
    ------------
    w_ :1d- array
        weights after fitting
    
    errors_: list
            Number of misclassifications in every epoch
            
    shuffle : bool (default:True)
            Shuffles training data every epoch.
            if True to prevent cycles.
    random_state : int (default: None)
            Set random state for shuffling
            and initiating the weights.
    """
    def __init__(self, eta=0.01, n_iter=10,
                     shuffle =True, random_state=None):
        self.eta= eta
        self.n_iter =n_iter
        self.w_initialized =False
        self.shuffle =shuffle
        
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        """ Fit training data
        Parameters
        ----------
        X : {array-like}, shape=[n_samples, n_features]
            Training vectors,
            where n_sample is the numbe rof samples and
            n_features is the number of featurs.
        y : array- like, shape=[n_samples]
            Target values.
            
        Returns
        -------
        self: object
        """
        self.initialize_weights(X.shape[1])
        self.cost_=[]
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y =self._shuffle(X,y)
            cost=[]
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost =sum(cost)/len(y)
            self.cist_.append(avg_cost)
        return self
        
        self.w_ = np.zeros(1+ X.shape[1])
        self.cost_ =[]
        
    def partial_fit(self, X, y):
        """ Fir training data without reinitializing the weights """
        if not self.w_initialized:
            self._initialize_weughts(X.shape[1])
        if y.ravel().shape[0] >1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_Weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        """ Shhuffle training data"""
        r=np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weighst to zero"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ Apply Adaline learning rule to update the weights """
        output = self.net_input(xi)
        error= (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost =0.5* error**2
        return cost
    
        
    def net_input(self, X):
        '''Calculate net input 
            transpose(w)*X '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation (self, X):
        '''compute linear activation'''
        return self.net_input(X)

    def predict(self, X):
        ''' return class label after unit step '''
        return np.where(self.activation(X) >= 0.0, 1, -1)
