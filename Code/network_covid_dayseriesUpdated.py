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
from numpy import inf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

plot_path = '../plots'
if not os.path.exists(plot_path):
    os.makedirs('plots')
os.chdir(plot_path)
plot_path = os.getcwd()

file_name = 'plotTotal'
if not os.path.exists(plot_path + '/' + file_name):
    os.makedirs(file_name)
    
plot_path =  plot_path + '/' + file_name + '/'   

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('red')
        else:
            cols.append('orange')
    return cols

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('red')
        else:
            cols.append('orange')
    return cols

def main():
    
    lastday=364 #125/129
    # df=pd.read_csv('../DatasetsDaySeriesCovidDataSetAgegroup.csv')
    df=pd.read_csv('../DatasetsDaySeriesCovidDataSet.csv')
    # df = df.sample(frac=1).reset_index(drop=True)
    # target = df[['Total_Deaths','west(0)/east(1)']]
    df=df.drop(['Cases_Per_Million','Area','Avg_Age','Std_dev'],axis=1)
    
    df_age=pd.read_csv('../Datasets/AgegroupCovidDataSet.csv')
    df_age = df_age[df_age['Landkreis'].isin(df['Landkreis'])].reset_index(drop=True)
    df['Einwohner(35-79)']=df_age['Einwohner(35-79)']        
    
    # df=df[df['Landkreis']!='Berlin']
    # df=df[df['Landkreis']!='Hamburg']
    # df=df[df['Landkreis']!='M\xc3\xbcnchen']
    
    df2 = pd.melt(df, id_vars=['Landkreis','LK_Einwohner','Cases','Deaths','Density','Income','<30','30-65','>65','Einwohner(35-79)','firstday','west(0)/east(1)'], 
                  var_name="Day", value_name="DailyCases")
    df2['Day'] = pd.to_numeric(df2['Day'],errors='coerce')
    df2=df2.sort_values(['Landkreis','Day']).reset_index(drop=True)
    df2['Relative(<30)']=df2['<30']/df2['LK_Einwohner']
    df2['Relative(30-65)']=df2['30-65']/df2['LK_Einwohner']
    df2['Relative(>65)']=df2['>65']/df2['LK_Einwohner']
    df2['RelativeEinwohner(35-79)']=df2['Einwohner(35-79)']/df2['LK_Einwohner']
    df2['LogDailycases']=np.log(df2['DailyCases'])
    df2['LogDailycases'][df2['LogDailycases']== -inf ]=0   
    
    df2['RelativeDailycases']=df2['DailyCases']/df2['LK_Einwohner']
    df2['RelativeLogDailycases']=np.log(df2['RelativeDailycases'])
    df2['RelativeLogDailycases'][df2['RelativeLogDailycases']== -inf ]= np.log(1.0/df2['LK_Einwohner'])   
    
    df2 = df2.sample(frac=1).reset_index(drop=True)
    # target = df2[['RelativeLogDailycases','west(0)/east(1)']]
    target = df2[['DailyCases','west(0)/east(1)']]
    target = target.values
    # features = df2[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]
    features = df2[['Density','Income','<30','30-65','>65','firstday','Day']]
    # features = df2[['Density','Income','RelativeEinwohner(35-79)','firstday','Day']]
    features = features.values
    
    df2=df2.sort_values(['Landkreis','Day']).reset_index(drop=True)
    # testfeatures = df2[['Density','Income','Relative(<30)','Relative(30-65)','firstday','Day']]    
    testfeatures = df2[['Density','Income','<30','30-65','>65','firstday','Day']]
    # testfeatures = df2[['Density','Income','RelativeEinwohner(35-79)','firstday','Day']]
    testfeatures = testfeatures.values
    # testtarget = df2[['RelativeLogDailycases','west(0)/east(1)']]
    testtarget=df2[['DailyCases','west(0)/east(1)']]
    testtarget = testtarget.values
    
    model = keras.models.Sequential()
    # model.add(keras.layers.Dropout(0.2,input_shape=(6,)))
    model.add(keras.layers.Dense(50,activation=None,input_dim=7,kernel_initializer='normal',kernel_constraint=max_norm(5)))
    for i in range(0,10):
        model.add(keras.layers.Dense(50,activation='elu',kernel_initializer='normal',kernel_constraint=max_norm(5)))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)
    
    # weights = model.get_weights()

    opt = keras.optimizers.Adam(lr=1e-4)
    # model.set_weights(weights)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)
    
    opt = keras.optimizers.Adam(lr=1e-5)
    # model.set_weights(weights)
    model.compile(optimizer=opt,loss='logcosh',metrics=['logcosh'])
    history = model.fit(x=features,y=target[:,:1],batch_size=100 ,epochs=15)#,validation_split=0.1)   
    
    # model_json = model.to_json()
    # with open("Totallog.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("TotalLog.h5")    

    model_json = model.to_json()
    with open("Plots/Total.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Plots/Total.h5")    
    
    history = history.history
    fig, ax1 = plt.subplots()
    ax1.plot(history['loss'],label='Training Loss')
    # ax1.plot(history['val_loss'],label='Val Loss')
    ax1.legend()
    fig.suptitle('Plot of Losses')
    
    test_loss, test_acc = model.evaluate(features, target[:,:1])
    
    # testtarget[:,0]=np.exp(testtarget[:,0])
    
    
    for i in range(0,df.shape[0]):
        target_predict = model.predict(x=testfeatures[i*lastday:(i+1)*lastday,:]) 
        # target_predict=np.exp(target_predict[:,0])  
        
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        fig, ax = plt.subplots()
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,6],testtarget[i*lastday:(i+1)*lastday,0],c='orange',label=r'\textbf{Actual}')
        ax.plot(testfeatures[i*lastday:(i+1)*lastday,6],target_predict,c='blue',label=r'\textbf{Predicted}')
        ax.set_xlabel(r'\textbf{Age Of the pandemic (Start-28-01-2020)}')
        # ax.set_xlabel("Age Of the pandamic (Start-02-02-2020)")
        ax.set_ylabel(r'\textbf{Total Cases}')
        ax.set_title(r'\textbf{Plot of Total Cases}')
        # ax.set_ylabel("Log of Total relative Cases")
        # ax.set_title("Log Plot of Total relative Cases")
        ax.legend()           
        filename=plot_path+str(int(df['west(0)/east(1)'].iloc[i]))+df['Landkreis'].iloc[i]+'.pdf'
        # filename='plotsLog/'+str(int(df['west(0)/east(1)'].iloc[i]))+df['Landkreis'].iloc[i]+'.png'        
        ax.figure.savefig(filename)


    
if __name__ == "__main__":
    main()