# logistic model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
EPS = 1e-8

np.random.seed(0)
# read files
x = pd.read_csv('./data/X_train.csv', index_col=0).astype(np.float).to_numpy()
y = pd.read_csv('./data/Y_train.csv', index_col=0).astype(np.int).to_numpy().flatten()
x_test = pd.read_csv('./data/X_test.csv',index_col=0).astype(np.float).to_numpy()

# === some useful function ===

sigmoid = lambda z: np.clip(1 / (1 + np.exp(-z)),EPS,1-EPS) #clip 函数

def normalize(x: np.ndarray):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + EPS)


def shuffle(x, y):
    """shuffle the dataset, often using MBGD

    Args:
        x ([np.ndarray]): [input dataset]
        y ([np.ndarray]): [label dataset]

    Returns:
        (x,y): shuffled input dataset and label dataset
    """
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return x[randomize], y[randomize]


def divideSet(x, y, ratio=0.25):
    p = int(len(x)*(1-ratio))
    return x[:p, :], x[p:, :], y[:p], y[p:]


def predict(x, w, is_bool = True):
    # use to binary classification
    if is_bool:
        return np.round(sigmoid(np.dot(x, w)))
    else:
        return sigmoid(np.dot(x, w))

def accurate(y, y_label):
    # calculate the accuracy
    return 1 - np.mean(np.abs(y - y_label))

# regularization / normalization
x = normalize(x)

# divide set
x_train, x_valid, y_train, y_valid = divideSet(x, y, 0.1)
train_size = len(x_train)
valid_size = len(x_valid)
print(f'training dataset size: {train_size}')
print(f'validation / devlopment dataset size: {valid_size}')

def cross_entrophy(y_pred, y_label):
    return -np.dot(y_label,np.log(y_pred+EPS)) - np.dot(1-y_label,np.log(1-y_pred+EPS))

def gradient(x, y_label, w):
    """
    Returns:
        (w_grad,b_grad): weight and bias gradient descent
    """
    return np.dot(x.T, sigmoid(np.dot(x, w)) - y_label)

#  === start train Mini-Batch Gradient decent ===

# MBGD variable config
max_iter = 20
batch_size = 16
eta = 0.2

# variables for plotting
train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []

# weights and coefficients
dim = x_train.shape[1] + 1
w = np.zeros((dim,))

step = 1    # use to decay learning rate

for epoch in range(max_iter):
    # shuffle dataset
    x_train, y_train = shuffle(x_train, y_train)
    for idx in range(int(np.floor(train_size / batch_size))):
        x_batch = x_train[idx*batch_size:(idx+1)*batch_size]
        y_batch = y_train[idx*batch_size:(idx+1)*batch_size]
        x_batch = np.concatenate((np.ones((len(x_batch),1)), x_batch), 1)

        w_grad = gradient(x_batch,y_batch,w)
        w = w - eta / np.sqrt(step) * w_grad
        step += 1

    # Compute loss and accuracy
    y_train_pred = predict(np.concatenate((np.ones((train_size,1)), x_train), 1),w,False)
    Y_train_pred = np.round(y_train_pred)
    train_accuracy.append(accurate(Y_train_pred, y_train))
    train_loss.append(cross_entrophy(y_train_pred, y_train) / train_size)

    y_valid_pred = predict(np.concatenate((np.ones((valid_size,1)), x_valid), 1),w,False)
    Y_valid_pred = np.round(y_valid_pred)
    valid_accuracy.append(accurate(Y_valid_pred,y_valid))
    valid_loss.append(cross_entrophy(y_valid_pred,y_valid) / valid_size)

print(f"train loss {round(train_loss[-1],2)}, training accuracy {round(train_accuracy[-1],2)}")
print(f"validation loss {round(valid_loss[-1],2)}, validation accuracy {round(valid_accuracy[-1],2)}")


# plot loss curve
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title("Loss Curve")
plt.legend(["train","validation"])
plt.show()

# plot accuracy curve
plt.plot(train_accuracy)
plt.plot(valid_accuracy)
plt.title("Accuracy Curve")
plt.legend(["train", "validation"])
plt.show()

# predict y
x_test = normalize(x_test)
y_test = predict(np.concatenate((np.ones((len(x_test),1)), x_test), 1),w)
rs = pd.DataFrame(y_test,columns=["label"]).astype(np.int)
rs.to_csv("ans2.csv")
print("over")