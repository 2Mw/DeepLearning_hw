import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

# preprocess
train_data = pd.read_csv('./train.csv').iloc[:, 3:]
train_data[train_data == 'NR'] = 0  # 解决未记录数值的问题
train_data = train_data.astype('float64')
train_data = train_data.to_numpy()

data_month = {}
for month in range(12):
    sample = np.empty([18, 480], dtype=float)
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = train_data[(day + month * 20) * 18: (month * 20 + day + 1) * 18, :]
    data_month[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty((12 * 471, 1), dtype=float)
for month in range(12):
    for i in range(471):
        x[month * 471 + i, :] = data_month[month][:, i:i + 9].reshape(1, -1)
        y[month * 471 + i, :] = data_month[month][9, i + 9]

# normalize
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(std_x)):
    if std_x[i] != 0:
        x[:, i] = (x[:, i] - mean_x[i]) / std_x[i]

# divide validation set
x_train_set = x[:math.floor(len(x) * 0.8), :]
y_train_set = y[:math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# start train
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1)
LEARNING_RATE = 100
ITER_TIMES = 1000
adagrad = np.zeros([dim, 1])
eps = 1e-10
loss, last_loss = 0, 0
loss_list = []
for t in range(ITER_TIMES):
    last_loss = loss
    loss = np.sqrt(np.sum((y - np.dot(x, w)) ** 2) / (471 * 12))
    loss_list.append(loss)
    gradient = 2 * np.dot(x.T, np.dot(x, w) - y)
    adagrad += gradient ** 2
    w = w - LEARNING_RATE * gradient / np.sqrt(adagrad + eps)
print("%2.f, %.5f" % (loss, loss - last_loss))
ts = pd.Series(loss_list)
ts.plot()
plt.title('loss curve')
plt.show()


# test data
t_data = pd.read_csv('test.csv', names=['id','items','0','1','2','3','4','5','6','7','8'], index_col=0)
testdata = t_data.iloc[:, 1:]
testdata[testdata == 'NR'] = 0
testdata = testdata.astype(float).to_numpy()
test_x = np.empty([240, 18 * 9])
for i in range(240):
    test_x[i, :] = testdata[i * 18:(i + 1) * 18, :].reshape(1, -1)
for i in range(len(std_x)):
    if std_x[i] != 0:
        test_x[:, i] = (test_x[:, i] - mean_x[i]) / std_x[i]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1)
res = np.dot(test_x, w).reshape(1, -1)

# output
r = pd.DataFrame(res.T,t_data.groupby('items').get_group('PM2.5').index,['value'])
r.to_csv('ans2.csv')
print('over')
