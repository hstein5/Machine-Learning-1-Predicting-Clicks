# -*- coding: utf-8 -*-
"""
Python Code for Neural Networks

@author: Ricky Chen, Yifei Ren, Halle Steinberg, Robert Wei
Team 4 
"""

######################################### Neural Network Models ########################################
#%% 
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from keras import backend as K
from keras import optimizers

NEpochs = 1000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.01)

def SetTheSeed(Seed):
    np.random.seed(Seed)
    rn.seed(Seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    tf.set_random_seed(Seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

# Read in the data   
TrainData = pd.read_csv('C:/Users/14702/Downloads/TrainDataForBCExample.csv',sep=',',header=0,quotechar='"')
ValData = pd.read_csv('C:/Users/14702/Downloads/ValDataForBCExample.csv',sep=',',header=0,quotechar='"')
ValData2 = pd.read_csv('C:/Users/14702/Downloads/ValDataForBCExample2.csv',sep=',',header=0,quotechar='"')
TestData = pd.read_csv('C:/Users/14702/Downloads/TestDataForBCExample.csv',sep=',',header=0,quotechar='"')

# split X and Y variables
Vars = list(TrainData)
YTr = np.array(TrainData[Vars[0]])
XTr = np.array(TrainData.loc[:,Vars[1:]])

Vars = list(ValData)
YVal = np.array(ValData[Vars[0]])
XVal = np.array(ValData.loc[:,Vars[1:]])

XTest = np.array(TestData)

Vars = list(ValData2)
YVal2 = np.array(ValData2[Vars[0]])
XVal2 = np.array(ValData2.loc[:,Vars[1:]])

# Rescale Training Data
XTrRsc = (XTr - XTr.min(axis=0))/XTr.ptp(axis=0)
XTrRsc.shape
XTrRsc.min(axis=0)
XTrRsc.max(axis=0)

# Rescale Validation Data. Need to use Training parameters to rescale.
XValRsc = (XVal - XTr.min(axis=0))/XTr.ptp(axis=0)
XValRsc.shape
XValRsc.min(axis=0)
XValRsc.max(axis=0)

#Rescale Validation 2 Data. Really should use Training parameters to rescale.
XValRsc2 = (XVal2 - XTr.min(axis=0))/XTr.ptp(axis=0)
XValRsc2.shape
XValRsc2.min(axis=0)
XValRsc2.max(axis=0)

# Rescale Test Data. Need to use Training parameters to rescale.
XTestRsc = (XTest - XTr.min(axis=0))/XTr.ptp(axis=0)
XTestRsc.shape
XTestRsc.min(axis=0)
XTestRsc.max(axis=0)


#%% Set up Neural Net Model - 4 layers of 4 neurons probability output
from keras.models import Sequential
from keras.layers import Dense, Activation

BCNN4 = Sequential()

BCNN4.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNN4.add(Dense(units=4,activation="relu",use_bias=True))
BCNN4.add(Dense(units=4,activation="relu",use_bias=True))
BCNN4.add(Dense(units=4,activation="relu",use_bias=True))
BCNN4.add(Dense(units=1,activation="sigmoid",use_bias=True))

BCNN4.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy'])

#%% Fit NN Model
FitHist = BCNN4.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=0)
print("Number of Epochs = "+str(len(FitHist.history['binary_crossentropy'])))
FitHist.history['binary_crossentropy'][-1]
FitHist.history['binary_crossentropy'][-10:-1]

#%% Make Predictions
YHatTr4 = BCNN4.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatTr4 = YHatTr4.reshape((YHatTr4.shape[0]),)

YHatVal4 = BCNN4.predict(XValRsc,batch_size=XValRsc.shape[0])
YHatVal4 = YHatVal4.reshape((YHatVal4.shape[0]),)


#%% Set up another Neural Net Model - 4 layers of 10 neurons probability output
from keras.models import Sequential
from keras.layers import Dense, Activation

BCNN10 = Sequential()

BCNN10.add(Dense(units=10,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNN10.add(Dense(units=10,activation="relu",use_bias=True))
BCNN10.add(Dense(units=10,activation="relu",use_bias=True))
BCNN10.add(Dense(units=10,activation="relu",use_bias=True))
BCNN10.add(Dense(units=1,activation="sigmoid",use_bias=True))

BCNN10.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy'])

#%% Fit NN Model
FitHist = BCNN10.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=0)
print("Number of Epochs = "+str(len(FitHist.history['binary_crossentropy'])))
FitHist.history['binary_crossentropy'][-1]
FitHist.history['binary_crossentropy'][-10:-1]

#%% Make Predictions
YHatTr10 = BCNN10.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo.
YHatTr10 = YHatTr10.reshape((YHatTr10.shape[0]),)

YHatVal10 = BCNN10.predict(XValRsc,batch_size=XValRsc.shape[0])
YHatVal10 = YHatVal10.reshape((YHatVal10.shape[0]),)


#%% Now try using softmax
from keras.models import Sequential
from keras.layers import Dense, Activation

BCNNsm = Sequential()

BCNNsm.add(Dense(units=4,input_shape=(XTrRsc.shape[1],),activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=4,activation="relu",use_bias=True))
BCNNsm.add(Dense(units=2,activation="softmax",use_bias=True))

BCNNsm.compile(loss='categorical_crossentropy', optimizer=Optimizer,metrics=['categorical_crossentropy'])

#%% Fit NN Model with Softmax
# Need to make YTr an n by 2 matrix
YTr = np.array([1-YTr,YTr]).transpose()

FitHist = BCNNsm.fit(XTrRsc,YTr,epochs=NEpochs,batch_size=BatchSize,verbose=0)
print("Number of Epochs = "+str(len(FitHist.history['categorical_crossentropy'])))
FitHist.history['categorical_crossentropy'][-1]
FitHist.history['categorical_crossentropy'][-10:-1]

# Make Predictions
YHatTrSM = BCNNsm.predict(XTrRsc,batch_size=XTrRsc.shape[0]) # Note: Not scaled, so not necessary to undo
YHatValSM = BCNNsm.predict(XValRsc,batch_size=XValRsc.shape[0]) # Note: Not scaled, so not necessary to undo


#%% write out data set
TrOutDF = pd.DataFrame(data={ 'YHatTr4': YHatTr4, 'YHatTr10': YHatTr10,'YHatTrSM': YHatTrSM[:,1] })
ValOutDF = pd.DataFrame(data={ 'YHatVal4': YHatVal4, 'YHatVal10': YHatVal10,'YHatValSM': YHatValSM[:,1] })

TrOutDF.to_csv('C:/Users/14702/Downloads/TrYHatFromBCNN.csv',sep=',',na_rep="NA",header=True,index=False)
ValOutDF.to_csv('C:/Users/14702/Downloads/ValYHatFromBCNN.csv',sep=',',na_rep="NA",header=True,index=False)

#%% Make Predictions on Validation data set 2
# according to best log loss, uese 4 layes of 4 neurons
YHatVal2 = BCNN4.predict(XValRsc2,batch_size=XValRsc2.shape[0])
YHatVal2 = YHatVal2.reshape((YHatVal2.shape[0]),)

# write out data set
ValOutDF2 = pd.DataFrame(data={ 'YHatVal2': YHatVal2})
ValOutDF2.to_csv('C:/Users/14702/Downloads/ValYHatFromBCNN2.csv',sep=',',na_rep="NA",header=True,index=False)

#%% Make Predictions on Test data set 
# according to best log loss, uese 4 layers of 4 neurons
YHatTest = BCNN4.predict(XTestRsc,batch_size=XTestRsc.shape[0])
YHatTest = YHatTest.reshape((YHatTest.shape[0]),)

# write out data set
TestOutDF = pd.DataFrame(data={ 'YHatTest': YHatTest})
TestOutDF.to_csv('C:/Users/14702/Downloads/TestYHatFromBCNN.csv',sep=',',na_rep="NA",header=True,index=False)