# -*- coding: utf-8 -*-
"""
Python Code for XGboost

@author: Ricky Chen, Yifei Ren, Halle Steinberg, Robert Wei
Team 4 
"""

############################################# XGboost ###########################################
#%% 
import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score 
from sklearn.metrics import make_scorer, mean_squared_error

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

Vars = list(ValData2)
YVal2 = np.array(ValData2[Vars[0]])
XVal2 = np.array(ValData2.loc[:,Vars[1:]])

XTest = np.array(TestData)


#%% Parameters tuning
# define log loss function as scoring 
def ll(YVal,ypred):
    ll = np.mean(YVal*np.log(ypred)+(1-YVal)*np.log(1-ypred))
    return(ll)

# set log loss as new scorer
new_scorer=make_scorer(ll, greater_is_better=False)

# tuning 'learning_rate'
param_test2 = {
 'learning_rate':[0.05,0.1,0.15,0.2,0.25,0.3]
}
gsearch = GridSearchCV(
    estimator = XGBClassifier(objective= 'binary:logistic',seed=27), 
    param_grid = param_test2, 
    scoring=new_scorer,
    cv=5)
gsearch.fit(XTr,YTr)
gsearch.best_params_  
# it turns out 'learning_rate' = 0.15 minimize log loss 

# tuning 'max_depth'
param_test2 = {
 'max_depth':[1,2,3,4,5,6,7,8,9,10]
}
gsearch2 = GridSearchCV(
    estimator = XGBClassifier(objective= 'binary:logistic', learning_rate=0.15 , seed=27), 
    param_grid = param_test2, 
    scoring=new_scorer,
    cv=5)
gsearch2.fit(XTr,YTr)
gsearch2.best_params_     
# it turns out 'max_depth' = 5 minimize log loss 

# tuning 'min_child_weight'
param_test2 = {
 'min_child_weight':[1,2,3,4,5,6,7,8,9,10]
}
gsearch2 = GridSearchCV(
    estimator = XGBClassifier(objective= 'binary:logistic', learning_rate=0.15 , max_depth = 5, seed=27), 
    param_grid = param_test2, 
    scoring=new_scorer,
    cv=5)
gsearch2.fit(XTr,YTr)
gsearch2.best_params_     
# it turns out 'min_child_weight' = 4 minimize log loss 

#%% train model and make prediction 
dtrain = xgb.DMatrix(data=XTr,label=YTr)
dval = xgb.DMatrix(data=XVal,label=YVal)

param = { 'objective': 'binary:logistic', 'learning_rate':0.15, 'max_depth':5,'min_child_weight':4}
num_round = 1500
evallist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist,early_stopping_rounds=80)

# make prediction on best round number 
ypredval = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
# the log loss is 0.40152
ll(YVal,ypredval)

# predict on another validation dataset
dval2 = xgb.DMatrix(data=XVal2,label=YVal2)
ypredval2 = bst.predict(dval2, ntree_limit=bst.best_ntree_limit)
# the log loss is 0.40083
ll(YVal2,ypredval2)

# predict test dataset clicks
dtest = xgb.DMatrix(XTest)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

# write out data set
TestOutDF = pd.DataFrame(data={ 'YHatTest': ypred})
TestOutDF.to_csv('C:/Users/14702/Downloads/TestYHatFromXGB.csv',sep=',',na_rep="NA",header=True,index=False)

