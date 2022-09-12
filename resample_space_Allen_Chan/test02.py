import numpy as np
from pyDOE import lhs
import torch
f_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X_f_test = np.arange(20).reshape(10, 2)
index = np.argsort(f_pred.flatten())
index = index[::-1]
num = int(0.3 * len(index))
#num = 200
index = index[:num]
resample = X_f_test[index]
X_f_retain = np.arange(10).reshape(5, 2)
X_f = np.vstack([X_f_retain, resample])

print(X_f)
a = np.vstack([X_f[0:3, :] , X_f[4:5, :]])
print(a)
i = 1
print("训练了%d步骤"%(i))
#x_val = np.linspace(-1., 1., 0.02)
time_list = np.linspace(0, 1, 10)
print(time_list)
print(len(time_list))


ub = np.array([1])
lb = np.array([0])
N_f = 10
X_f_test = 0 + (1 - 0) * lhs(1, N_f)
print(X_f_test.shape)
x_val = np.linspace( -1, 1, 10)
print(x_val.shape)

def data_iter( batch_size, features, labels=None):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        if labels == None:
            if features[batch_indices].ndim == 1:
                yield features[batch_indices].reshape(-1, 2)
                return 
            yield features[batch_indices]
        else:
            yield features[batch_indices], labels[batch_indices]


a = [1, 2, 3]
b = np.array(a)
print(b)
a = np.linspace(-1, 1, 3)[:, None]
print(a.shape)
j = 1
print('第%d次空间重采样'%(j+1)) 