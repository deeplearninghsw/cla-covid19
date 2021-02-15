# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:27:20 2020

@author: Ajay N R
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def genData(nSamples, withNoise=True,classified=False,frac=0.5):
    
    
    sigmoid = lambda x: 1/(1+np.exp(-x))
    
    X = np.arange(0, 1, 1./nSamples)
    Y = np.arange(0, 1, 1./nSamples)
    X, Y = np.meshgrid(X, Y)
    
    a1, b1 = 1,1
    Z1 = X*Y*(a1*X**2+b1*Y**2)
    Z1 = sigmoid(Z1)
    
    a2, b2 = 2,2
    Z2 = X*Y*(a2*X+b2*Y)
    Z2 = sigmoid(Z2)

    if withNoise:
        sigma = 0.05
        
        noise = sigma*np.random.randn(Z1.shape[0],Z1.shape[1])
        Z1 += noise
        Z1_nonoise=Z1-noise
        
        noise = sigma*np.random.randn(Z2.shape[0],Z2.shape[1])
        Z2 += noise
        Z2_nonoise=Z2-noise

    if classified:
        return X, Y, Z1, Z2
    else:
        X=X.ravel();
        Y=Y.ravel();
        Z1=Z1.ravel();
        Z2=Z2.ravel();
        
        Z1, Z2 = np.stack((Z1,np.zeros(Z1.size)),axis=-1), np.stack((Z2,np.ones(Z2.size)),axis=-1)
        
        ix1 = np.random.choice(X.size, int(np.round(frac*X.size)), replace=False)
        trainData = np.hstack([X[ix1,np.newaxis],Y[ix1,np.newaxis],Z1[ix1,:]])
        ix2 = [z for z in range(X.size) if not z in ix1]
        trainData = np.vstack([trainData,np.hstack([X[ix2,np.newaxis],Y[ix2,np.newaxis],Z2[ix2,:]])])
        trainData = trainData[trainData[:,0].argsort()]           
        return trainData,Z1_nonoise,Z2_nonoise
    
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('orange')
        else:
            cols.append('red')
    return cols



if __name__ == "__main__":
    classified = False
    n=100
    if classified:        
        X, Y, Z1, Z2 = genData(nSamples=n,withNoise=True,classified=classified)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z1, rstride=8, cstride=8, alpha=0.3,cmap=cm.coolwarm)
        ax.plot_surface(X, Y, Z2, rstride=8, cstride=8, alpha=0.3,cmap=cm.coolwarm)
        
    else:
        trainData,Z1_nonoise,Z2_nonoise  = genData(nSamples=n,frac=0.8) 
        X=trainData[:,0]
        X=np.reshape(X,(-1,n)).T
        Y=X.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cols = pltcolor(trainData[:,-1])
        ax.scatter(trainData[:,0], trainData[:,1], trainData[:,2],c=cols,s=1)
        ax.plot_surface(X, Y, Z1_nonoise,color='orange')
        ax.plot_surface(X, Y, Z2_nonoise,color='red')
        
        
        
