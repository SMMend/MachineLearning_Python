# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:58:46 2020

@author: sybil
Implementing an Adaptive Linear Neuron
"""
import numpy as np
import matplotlib.pyplot as plt

class AdalineGD(object):
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
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta= eta
        self.n_iter =n_iter
        
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
        
        self.w_ = np.zeros(1+ X.shape[1])
        self.cost_ =[]
        
        for i in range(self.n_iter):
            output =self.net_input(X)
            errors =(y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] +=self.eta * errors.sum()
            cost =(errors **2).sum() /2.0
            self.cost_.append(cost)
        return self
    
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

fig, ax=plt.subplots(nrows=1, ncols=2,figsize=(8,4))
#figure 1
ada1 = AdalineGD(n_iter =10, eta=0.01).fit(X, y)
print(ada1.cost_)
ax[0].plot(range(1, len(ada1.cost_)+1),
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline-Learning rate 0.01')
#figure 2
ada2 = AdalineGD(n_iter =10, eta= 0.0001).fit(X,y)
print(ada2.cost_)
ax[1].plot(range(1, len(ada2.cost_)+1),
           ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()    
            
    
