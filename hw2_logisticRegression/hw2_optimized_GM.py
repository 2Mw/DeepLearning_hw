# Probabilistic generative model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv('./data/X_train.csv', index_col=0).astype(np.float).to_numpy()
y = pd.read_csv('./data/Y_train.csv', index_col=0).astype(np.int).to_numpy().flatten()
x_test = pd.read_csv('./data/X_test.csv', index_col=0).astype(np.float).to_numpy()
EPS = 1e-8

dim = x.shape[1]


def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + EPS)


x = normalize(x)

x_train_0 = np.array([a for a, b in zip(x, y) if b == 0])
x_train_1 = np.array([a for a, b in zip(x, y) if b == 1])

mean_0 = np.mean(x_train_0, axis=0)
mean_1 = np.mean(x_train_1, axis=0)

N1 = x_train_0.shape[0]
N2 = x_train_1.shape[0]

cov_0 = np.zeros((dim, dim))
cov_1 = np.zeros((dim, dim))

for i in x_train_0:
    cov_0 += np.dot(np.transpose([i - mean_0]), [i - mean_0]) / N1
for i in x_train_1:
    cov_1 += np.dot(np.transpose([i - mean_1]), [i - mean_1]) / N2

# shared covariance. Use weighted average of individual in-class covariance.
cov = (cov_0 * N1 + cov_1 * N2) / (N1 + N2)

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(N1 / N2)

y_train_pred = 1 - np.round(1 / (1 + np.exp(-(np.dot(x, w) + b))))
print(f'Training accuracy: {1 - np.mean(np.abs(y - y_train_pred))}')

x_test = normalize(x_test)
predict = 1 - np.round(1 / (1 + np.exp(-(np.dot(x_test, w) + b))))
rs = pd.DataFrame(predict,columns=["label"]).astype(np.int)
rs.to_csv("ans3.csv")
print("over")