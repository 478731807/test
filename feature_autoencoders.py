#coding=utf-8
#######使用栈式自动编码器(AutoEncoder)提取高级特性######
#激活函数- GELU（高斯误差)。我们将使用GELU作为自动编码器
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

batch_size = 64
n_batches = dataset_TI_df.shape[0]/batch_size
VAE_data = dataset_TI_df.values
num_training_days=1460
train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days,2:-1]},
                               label={'label': VAE_data[:num_training_days, 1]}, batch_size = batch_size)
test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:,2:-1]},
                              label={'label': VAE_data[num_training_days:,1]}, batch_size = batch_size)

class VAE(gluon.HybridBlock):
   def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784,batch_size=100, act_type='relu', **kwargs):
      self.soft_zero = 1e-10
      self.n_latent = n_latent
      self.batch_size = batch_size
      self.output = None
      self.mu = None
      super(VAE, self).__init__(**kwargs)

      with self.name_scope():
         self.encoder = nn.HybridSequential(prefix='encoder')

         for i in range(n_layers):
            self.encoder.add(nn.Dense(n_hidden, activation=act_type))
         self.encoder.add(nn.Dense(n_latent * 2, activation=None))

         self.decoder = nn.HybridSequential(prefix='decoder')
         for i in range(n_layers):
            self.decoder.add(nn.Dense(n_hidden, activation=act_type))
         self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

   def hybrid_forward(self, F, x):
      h = self.encoder(x)
      # print(h)
      mu_lv = F.split(h, axis=1, num_outputs=2)
      mu = mu_lv[0]
      lv = mu_lv[1]
      self.mu = mu

      eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
      z = mu + F.exp(0.5 * lv) * eps
      y = self.decoder(z)
      self.output = y

      KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
      logloss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
      loss = -logloss - KL
      return loss
n_hidden=400 # neurons in each layer
n_latent=2
n_layers=3  # num of dense layers in encoder and decoder respectively
n_output=VAE_data.shape[1]-1-2
net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size, act_type='relu')
net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})
print(net)
n_epoch = 150
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []
for epoch in range(n_epoch):
    epoch_loss = 0
    epoch_val_loss = 0

    train_iter.reset()
    test_iter.reset()

    n_batch_train = 0
    for batch in train_iter:
        n_batch_train +=1
        data = batch.data[0].as_in_context(mx.cpu())

        with autograd.record():
            loss = net(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()

    n_batch_val = 0
    for batch in test_iter:
        n_batch_val +=1
        data = batch.data[0].as_in_context(mx.cpu())
        loss = net(data)
        epoch_val_loss += nd.mean(loss).asscalar()

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss)
    validation_loss.append(epoch_val_loss)

    """if epoch % max(print_period, 1) == 0:
        print('Epoch {}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))"""
end = time.time()
print('Training completed in {} seconds.'.format(int(end-start)))




