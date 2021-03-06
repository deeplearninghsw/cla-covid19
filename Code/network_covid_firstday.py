#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:11:01 2020

@author: nihal
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('red')
        else:
            cols.append('orange')
    return cols

def main():

    df=pd.read_csv('../Datasets/FirstdayCovidDataSet.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    # target = df[['Total_Deaths','west(0)/east(1)']]
    df['age']=118-df['firstday']
    target = df[['Cases','west(0)/east(1)']]
    target = target.values
    features = df[['Density','Income','<30','30-65','>65','age']]
    features = features.values
    
    model = keras.models.Sequential()
    
    model.add(keras.layers.Dense(100,activation='elu',input_dim=6,kernel_initializer='normal',kernel_constraint=max_norm(5)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5))) 
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
    
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=16 ,epochs=25,validation_split=0.1)
    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')
    
    test_loss, test_acc = model.evaluate(features, target[:,:1])

    target_predict = model.predict(x=features) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cols = pltcolor(target[:,-1]) 
    ax.scatter(features[:,1],features[:,5],target[:,0],c=cols)
    ax.scatter(features[:,1],features[:,5],target_predict)

    
if __name__ == "__main__":
    main()
