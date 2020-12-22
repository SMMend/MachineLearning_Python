# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:03:17 2020

@author: sybil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron(object):
    """ Perceptron classifier
    
    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
            Passes over the training dataset.
            
    Attributes
    ----------
    w_ : 1d-array
        weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """
    
    def __init__(self, eta=0.01, n_iter =10):
        self.eta=eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        y : array-like, shape =[n_samples]
            Target values.
            
        Returns
        -------
        self : object
        """
        #shape[0] will give the number of rows in an array. 
        #If you will type x.shape[1] , it will print out the number of columns 
        # we add 1 for w0
        self.w_ =np.zeros(1 + X.shape[1])
        #print("X shape", X.shape[1])
        #print("w", self.w_)
        self.errors_ =[]
        
        # we put underscore when we don't are about the index values
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update =self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    #computes w^T*X
    def net_input(self, X):
        """Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
 
#reading in the data    
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.tail()

#extract the first 100 class labels that correspond to the 50 iris-setosa and
#50 iris-versicolor

# obraining the 1st 100 rows of column 4
#The iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position.
#iloc” in pandas is used to select rows and columns by number, in the order that they appear in the data frame. 
#The Pandas loc indexer can be used with DataFrames for two different use cases:
#a.) Selecting rows by label/index
#b.) Selecting rows with a boolean / conditional lookup

y=df.iloc[0:100, 4].values
y=np.where(y=='Iris-setosa', -1, 1)

#obtaining the 1st 100 observations of the first 2 columns 0, 1
X=df.iloc[0:100, [0,2]].values

# scatter plot 
#plot the setosa values
plt.scatter(X[:50, 0], X[:50,1], color ='red', marker='o', label='setosa')
#plot the versicolor values
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()


'''training the perceptron algorithm'''
ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) +1), ppn.errors_ , marker= 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


#meshgrid: https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution =0.02):
    #setup marker generator andcolor map
    markers=('s', 'x', 'o', '^', 'v')
    colors=('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    cmap= ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min, x1_max =X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max =X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2=np.meshgrid(np.arange(x1_min, x2_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z= classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print("Z", Z)
    Z= Z.reshape(xx1.shape)
    print("Z1", Z)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()    