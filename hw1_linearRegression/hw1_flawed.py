import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# preprocess
data = pd.read_csv('train.csv', header=0)
grouped = data.groupby(data.columns[2])
pm2_5 = grouped.get_group('PM2.5').loc[:, '0':'23']
pm2_5 = pm2_5.astype('float64')
inp = pm2_5.loc[:, '0':'8'].to_numpy()
ans = pm2_5.loc[:, '9'].to_numpy()
m = ans.shape[0]

# model
ALPHA = 1e-4
GAP = 1e-4
b = 0
w = np.array([0 for i in range(9)])
y = lambda x: np.dot(x, w) + b
out = y(inp)
loss = np.sum((ans - out) ** 2) / (2*m)
last_loss = loss
loss_list = []

while True:
    der_bias = - np.sum(ans - out) / m
    der_weight = -sum((inp.T*(ans-out)).T)/m
    w = w - ALPHA * der_weight
    b = b - ALPHA * der_bias
    # 重新计算
    out = y(inp)
    loss = np.sum((ans - out) ** 2) / (2*m)
    if abs(loss-last_loss) < GAP:
        break
    if loss > last_loss:
        print('not')
        break
    last_loss = loss
    loss_list.append(loss)

print(f'train over, loss is %.2f' % loss)

# plot loss curve

ts = pd.Series(loss_list)
ts.plot()
plt.title('Loss Curve')
plt.show()

# read test.csv

test = pd.read_csv('test.csv',names=['id','items','0','1','2','3','4','5','6','7','8'], index_col=0)
test_data = test.groupby('items').get_group('PM2.5')
test_input = test_data.loc[:,'0':'8'].astype('float64').to_numpy()
test_output = y(test_input)
df = pd.DataFrame(test_output,test_data.index,['value'])
df.to_csv('ans.csv')
print('输出文件完毕')


