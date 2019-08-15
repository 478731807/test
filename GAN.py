#coding=utf-8
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
from feature_autoencoders import *
#使用LSTM作为时间序列生成器，CNN作为鉴别器
#1. Metropolis-Hastings GAN

#2. Wasserstein GAN

##  LSTM
# 一个LSTM层有k个输入单元（变量）和n个隐藏单元（自定），一个密集层有1个输出(价格)。初始化器是Xavier，我们将使用L1损耗
#使用Adam（学习率为0.01）作为优化器
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

gan_num_features = dataset_TI_df.shape[1]-2
sequence_length = 17
class RNNModel(gluon.Block):
    def __init__(self, num_embed, num_hidden, num_layers, bidirectional=False,
                 sequence_length=sequence_length, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.rnn = rnn.LSTM(num_hidden, num_layers, input_size=num_embed,
                                bidirectional=bidirectional, layout='TNC')

            self.decoder = nn.Dense(1, in_units=num_hidden)

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


lstm_model = RNNModel(num_embed=gan_num_features, num_hidden=500, num_layers=1)
lstm_model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
trainer = gluon.Trainer(lstm_model.collect_params(), 'adam', {'learning_rate': .01})
loss = gluon.loss.L1Loss()

print(lstm_model)


############学习率###########
class TriangularSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class CyclicalSchedule():
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = iteration - idx + cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        return schedule(cycle_offset) * self.magnitude_decay ** cycle_idx

schedule = CyclicalSchedule(TriangularSchedule, min_lr=0.5, max_lr=2, cycle_length=500)
iterations=100
plt.plot([i+1 for i in range(iterations)],[schedule(i) for i in range(iterations)])
plt.title('Learning rate for each epoch')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()



num_fc = 512
# CNN
cnn_net = gluon.nn.Sequential()
with net.name_scope():
    # Add the 1D Convolutional layers
    cnn_net.add(gluon.nn.Conv1D(32, kernel_size=5, strides=2))
    cnn_net.add(nn.LeakyReLU(0.01))
    cnn_net.add(gluon.nn.Conv1D(64, kernel_size=5, strides=2))
    cnn_net.add(nn.LeakyReLU(0.01))
    cnn_net.add(nn.BatchNorm())
    cnn_net.add(gluon.nn.Conv1D(128, kernel_size=5, strides=2))
    cnn_net.add(nn.LeakyReLU(0.01))
    cnn_net.add(nn.BatchNorm())

    # Add the two Fully Connected layers
    cnn_net.add(nn.Dense(220, use_bias=False), nn.BatchNorm(), nn.LeakyReLU(0.01))
    cnn_net.add(nn.Dense(220, use_bias=False), nn.Activation(activation='relu'))
    cnn_net.add(nn.Dense(1))

