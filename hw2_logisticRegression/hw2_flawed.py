import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
import pandas as pd
import math
import warnings

warnings.filterwarnings("ignore")
print(__file__)
# load train files
x_csv = pd.read_csv("./data/X_train.csv", header=0, index_col=0).astype(np.float)
y_csv = pd.read_csv('./data/Y_train.csv', header=0, index_col=0).astype(np.float)
x = x_csv.to_numpy()
y = y_csv.to_numpy()

sigmoid = lambda z: 1 / (1 + np.exp(-z))

# regularization
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
for i in range(len(x_mean)):
    if x_std[i] != 0:
        x[:, i] = (x[:, i] - x_mean[i]) / x_std[i]

# divide set. Cross validation
x_train = x[:math.floor(len(x) * 0.8), :]
y_train = y[:math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# start train

iterate_times = 1000
eta = 5
dim = x_train.shape[1] + 1
w = np.zeros([dim,1])
adagrad = np.zeros([dim,1])
eps=1e-10
x_train = np.concatenate((np.ones([x_train.shape[0], 1]), x_train), axis=1)
loss_list = []

while iterate_times != 0:
    f = sigmoid(np.dot(x_train,w))
    # cross entropy
    loss = -np.sum((y_train*np.log(f+eps)+(1-y_train)*np.log(1-f+eps)))
    gradient = 2 * np.dot(x_train.T, np.dot(x_train, w) - y_train)
    adagrad += gradient ** 2
    w = w - eta * gradient / np.sqrt(adagrad+eps)
    iterate_times -= 1
    loss_list.append(loss)
    print(loss)

np.save('w.npy',w)

# paint the loss curve
ts = pd.Series(loss_list)
ts.plot()
plt.title("loss curve")
plt.show()

w = np.load('w.npy')

# train over, load testing data
x_test_csv = pd.read_csv('X_test.csv').astype(np.float)
x = x_test_csv.to_numpy()

# test dataset regularization

x_test_mean = np.mean(x, axis=0)
x_test_std = np.std(x, axis=0)
for i in range(len(x_test_mean)):
    if x_test_std[i] != 0:
        x[:, i] = (x[:, i] - x_test_mean[i]) / x_test_std[i]

res = sigmoid(np.dot(x, w)).reshape(1, -1)
rs_df = pd.DataFrame(res.T,columns=['label']).astype(np.int)
rs_df = np.round(rs_df).astype(np.int)
rs_df.to_csv('ans.csv')


