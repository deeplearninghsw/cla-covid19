#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:50:08 2020

@author: nihal
"""


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

plot_path = '../plots'
if not os.path.exists(plot_path):
    os.makedirs('plots')
os.chdir(plot_path)
plot_path = os.getcwd()



def genData(n,noise):
    np.random.seed(0)
    f1 = lambda x: ((x-4)*(x+4))**2+np.random.normal(0,noise,x.size)
    
    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n):
        x[i] = np.sort(-6+ np.random.rand(1) *12)
        y[i] = f1(x[i])
        
    
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    
    trainData = np.hstack((x,y))
    
    return trainData 


trainData = genData(1000,2)
ix = np.random.choice(trainData.shape[0],int(trainData.shape[0]*0.2),replace=False)
testData = trainData[ix,:]
trainData = np.delete(trainData,ix,axis=0)


model = keras.models.Sequential()
model.add(keras.layers.Dense(50,activation=None,input_dim=1))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(50,activation='elu'))
model.add(keras.layers.Dense(1))

opt = keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=opt,loss='logcosh',metrics=['mae'])

val_split = 0.2
batch_size = 32
epochs = 50

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', )

hist = model.fit(x=trainData[:,0],y=trainData[:,1],batch_size = batch_size, epochs = epochs,validation_split=val_split)
                 
predData = model.predict(testData[:,0])

history = hist.history
fig, ax1 = plt.subplots()
ax1.plot(history['loss'],label='Training Loss')
ax1.plot(history['val_loss'],label='Val Loss')
ax1.legend()
fig.suptitle('Plot of Losses')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax2 = plt.subplots()
ax2.scatter(testData[:,0],testData[:,1],marker='.',label=r'\textbf{Test Data}')
ax2.scatter(testData[:,0],predData,marker='.',label=r'\textbf{Pred Data}')

ax2.legend()
ax2.set_xlabel(r'\textbf{X}', fontsize=11)
ax2.set_ylabel(r'\textbf{Y}', fontsize=11)
fig.suptitle(r'\textbf{Network Prediction (1D Case)}', fontsize=11)
plt.show()

ax2.figure.savefig(plot_path + "/model1_1.pdf", bbox_inches='tight')