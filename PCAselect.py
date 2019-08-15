#coding=utf8

##########################   columns = [c for c in columns if c not in ["test", 'date']]       ####################
from utils import *
import time
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
context = mx.cpu()
model_ctx=mx.cpu()
mx.random.seed(1719)
def parser(x):
   return datetime.datetime.strptime(x,'%Y/%m/%d')
dataset_ex_df = pd.read_csv('gzmt.csv', header=0, parse_dates=[0], date_parser=parser)
def get_technical_indicators(dataset):
   # Create 7 and 21 days Moving Average
   dataset['ma7'] = dataset['price'].rolling(window=7).mean()
   dataset['ma21'] = dataset['price'].rolling(window=21).mean()
   # Create MACD
   dataset['26ema'] = pd.ewma(dataset['price'], span=26)
   dataset['12ema'] = pd.ewma(dataset['price'], span=12)
   dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
   # Create Bollinger Bands
   dataset['20sd'] = pd.stats.moments.rolling_std(dataset['price'], 20)
   dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
   dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
   # Create Exponential moving average
   dataset['ema'] = dataset['price'].ewm(com=0.5).mean()
   # Create Momentum
   dataset['momentum'] = dataset['price'] - 1
   dataset['log-momentum']=np.log(dataset['momentum'])
   return dataset
dataset_TI_df = get_technical_indicators(dataset_ex_df)

######## Eigen portfolio with PCA ##########
# We want the PCA to create the new components to explain 80% of the variance
#vae_added_df = mx.nd.array(dataset_TI_df.iloc[20:, 2:-1].values)
vae_added_df = np.array(dataset_TI_df.iloc[20:, 2:-1].values)
pca = PCA(n_components=0.9)
x_pca = StandardScaler().fit_transform(vae_added_df)
principalComponents = pca.fit(x_pca)
print(principalComponents.n_components_)
print(principalComponents.explained_variance_ratio_)
print(principalComponents.get_params)