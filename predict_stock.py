#coding=utf8
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
## 基本图
# plt.figure(figsize=(14, 5), dpi=100)
# plt.plot(dataset_ex_df['Date'], dataset_ex_df['price'], label='Maotai')
# plt.vlines(datetime.date(2016,4, 20), 0, 800, linestyles='--', colors='gray', label='Train/Test data cut-off')
# plt.xlabel('Date')
# plt.ylabel('RMB')
# plt.title('Figure 2: GZMT stock price')
# plt.legend()
# plt.show()

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

# def plot_technical_indicators(dataset, last_days):
#    plt.figure(figsize=(16, 10), dpi=100)
#    shape_0 = dataset.shape[0]
#    xmacd_ = shape_0 - last_days
#    dataset = dataset.iloc[-last_days:, :]
#    x_ = range(3, dataset.shape[0])
#    x_ = list(dataset.index)
#    # Plot first subplot
#    plt.subplot(2, 1, 1)
#    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
#    plt.plot(dataset['price'], label='Closing Price', color='b')
#    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
#    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
#    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
#    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
#    plt.title('Technical indicators for GZMT - last {} days.'.format(last_days))
#    plt.ylabel('GZMT')
#    plt.legend()
#    # Plot second subplot
#    plt.subplot(2, 1, 2)
#    plt.title('MACD')
#    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
#    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
#    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
#    plt.plot(dataset['log-momentum'], label='Momentum', color='b', linestyle='-')
#    plt.legend()
#    plt.show()
# plot_technical_indicators(dataset_TI_df, 400)


###########基本面分析############
#将对所有相关的每日新闻进行情绪分析。最后使用sigmoid，结果将在0到1之间。得分越接近0，负面消息就越多（接近1表示正面情绪）
# 对于每一天，我们将创建平均每日分数（作为0到1之间的数字），并将其添加为一个特征


##BERT
import bert


######用于趋势分析的傅里叶变换######
#我们使用傅里叶变换的目的是提取长期和短期的趋势，所以我们将使用含有3、6和9个分量的变换。可以推断，包含3个组件的转换是长期趋势
data_FT = dataset_ex_df[['Date', 'price']]
close_fft = np.fft.fft(np.asarray(data_FT['price'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
# plt.figure(figsize=(14, 7), dpi=100)
# fft_list = np.asarray(fft_df['fft'].tolist())
# for num_ in [3, 6, 9, 100]:
#     fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
#     plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
# plt.plot(data_FT['price'],  label='Real')
# plt.xlabel('Days')
# plt.ylabel('RMB')
# plt.title('Figure 3: GZMT (close) stock prices & Fourier transforms')
# plt.legend()
# plt.show()


########用于降噪数据的另一种技术是调用小波(wavelets)。
# 小波和傅里叶变换给出了相似的结果所以我们只使用傅里叶变换( Fourier transform)
######
# from collections import deque
# items = deque(np.asarray(fft_df['absolute'].tolist()))
# items.rotate(int(np.floor(len(fft_df)/2)))
# plt.figure(figsize=(10, 7), dpi=80)
# plt.stem(items)
# plt.title('Figure 4: Components of Fourier transforms')
# plt.show()

######ARIMA是一种预测时间序列数据的方法。虽然ARIMA不能作为我们的最终预测，但我们将使用它作为一种技术来稍微降低库存的噪声，
# 并（可能）提取一些新的模式或特性
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
from sklearn.metrics import mean_squared_error
series = data_FT['price']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
# print(model_fit.summary())
#
# from pandas.plotting import autocorrelation_plot     #自相关图
# autocorrelation_plot(series)
# plt.figure(figsize=(10, 7), dpi=80)
# plt.show()


X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)

# Plot the predicted (from ARIMA) and real prices
# plt.figure(figsize=(12, 6), dpi=100)
# plt.plot(test, label='Real')
# plt.plot(predictions, color='red', label='Predicted')
# plt.xlabel('Days')
# plt.ylabel('RMB')
# plt.title('Figure 5: ARIMA model on GZMT stock')
# plt.legend()
# plt.show()


#########我们将通过ARIMA使用预测价格作为LSTM的输入特征###########
##统计检查：异方差，多重共线性，序列相关
#异方差：例如误差项随着数据点（沿x轴）的增长而增长
# 多重共线性是指错误项（也称为残差）相互依赖
# 序列相关性是指一个数据（特征）是另一个特征的公式（或完全不相关）


##利用XGBoost找出特性重要性
def get_feature_importance_data(data_income):
   data = data_income.copy()
   y = data['price']
   X = data.iloc[:, 2:]
   train_samples = int(X.shape[0] * 0.65)
   X_train = X.iloc[:train_samples]
   X_test = X.iloc[train_samples:]
   y_train = y.iloc[:train_samples]
   y_test = y.iloc[train_samples:]
   return (X_train, y_train), (X_test, y_test)
# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)  #调参
xgbModel = regressor.fit(X_train_FI,y_train_FI, eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
# plt.figure(1)
# plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
# plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
# plt.xlabel('Iterations')
# plt.ylabel('RMSE')
# plt.title('Training Vs Validation Error')
# plt.legend()
# plt.show()

print(xgbModel.feature_importances_)
fig = plt.figure(figsize=(8,8))
plt.xticks(rotation=45)
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
plt.title('Figure 6: Feature importance of the technical indicators.')
plt.show()



