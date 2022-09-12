import sys
import torch
from collections import OrderedDict

from torch.nn.modules import module

sys.path.insert(0, 'AC_minibatch_test1/Utilities')
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import copy
np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


learning_rate = 9e-4
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X_u, u, X_b_l, X_b_r, X_f, layers, lb, ub, X_f_test):

        # 两个角点
        self.lb = torch.tensor(lb).float().to(device)  # [-1,0]
        self.ub = torch.tensor(ub).float().to(device)  # [1, 1]

        # 训练数据集
        self.X_f = X_f
        self.X_f_retain = X_f.copy()
        
        # x_u、t_u:初边值的坐标
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)

        # 初值点的函数值
        self.u = torch.tensor(u).float().to(device)

        # 训练集中数据点的时间、空间坐标
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)

        # 边值采样点
        self.bl = torch.tensor(X_b_l[:, 0:1], requires_grad=True).float().to(device)
        self.br = torch.tensor(X_b_r[:, 0:1], requires_grad=True).float().to(device)
        self.bt = torch.tensor(X_b_l[:, 1:2], requires_grad=True).float().to(device)
        
        # 重采样点:在X_f_test上评估网络，然后从中挑出loss_f较大的点作为重采样点
        self.x_test = torch.tensor(X_f_test[:, 0:1], requires_grad=True).float().to(device)
        self.t_test = torch.tensor(X_f_test[:, 1:2], requires_grad=True).float().to(device)
        self.X_f_test = X_f_test
        
        # 神经网络
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        # 优化器
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)
        self.optimizer_LBFGS = torch.optim.LBFGS(self.dnn.parameters(),
                                           max_iter=200,
                                           tolerance_grad=1.e-8,
                                           tolerance_change=1.e-12,
                                           history_size=50)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_Adam,
                                   step_size=100,
                                   gamma=0.65)
        
        # 迭代次数
        self.iter = 0
        self.epoch_adam = 100
        self.epoch_lbfgs = 300
        
    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def data_iter(self, batch_size, features, labels=None):
        '''
        数据迭代器：
        给定数据集合features，每次返回batch_size个数据
        '''
        num_examples = len(features)
        indices = list(range(num_examples))
        np.random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(
                indices[i:min(i + batch_size, num_examples)]
            )
            if labels == None:
                yield features[batch_indices]
            else:
                yield features[batch_indices], labels[batch_indices]

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - 0.0001 * u_xx + 5 * u ** 3 - 5 * u
        return f

    def loss_func(self):
        # 清除梯度，计算损失，返回损失
        u_pred = self.net_u(self.x_u, self.t_u)

        ubl_pred = self.net_u(self.bl, self.bt)

        ubr_pred = self.net_u(self.br, self.bt)

        ubl_x = torch.autograd.grad(
            ubl_pred, self.bl,
            grad_outputs=torch.ones_like(ubl_pred),
            retain_graph=True,
            create_graph=True
        )[0]

        ubr_x = torch.autograd.grad(
            ubr_pred, self.br,
            grad_outputs=torch.ones_like(ubl_pred),
            retain_graph=True,
            create_graph=True
        )[0]

        loss_b = torch.mean((ubl_pred - ubr_pred) ** 2 + (ubl_x - ubr_x) ** 2)
  
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)

        # f(t, x)的点的集合，想让f_pred越小越好
        loss_f = torch.mean(f_pred ** 2)

        loss = 100 * loss_u + loss_f + loss_b
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
                self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def loss_func_batch(self, x_f, t_f):
        '''
        batch_size个点的loss_f,
        所有边界采样点的loss_b
        所有初值采样点的loss_u
        '''
        self.optimizer_Adam.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        ub1 = self.net_u(self.bl, self.bt)
        ub2 = self.net_u(self.br, self.bt)

        uxb1 = torch.autograd.grad(
            ub1, self.bl,
            grad_outputs=torch.ones_like(ub2),
            retain_graph=True,
            create_graph=True
        )[0]

        uxb2 = torch.autograd.grad(
            ub2, self.br,
            grad_outputs=torch.ones_like(ub2),
            retain_graph=True,
            create_graph=True
        )[0]
        
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_b = torch.mean((ub1 - ub2) ** 2 + (uxb1 - uxb2) ** 2)
        
        x_f = torch.tensor(x_f, requires_grad=True).float().to(device)
        t_f = torch.tensor(t_f, requires_grad=True).float().to(device)

        f_pred = self.net_f(x_f, t_f)

        # f(t, x)的点的集合，想让f_pred越小越好
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_f + 100 * loss_u + loss_b
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_b: %.5e, Loss_f: %.5e' % (
                    self.iter, loss.item(), loss_u.item(), loss_b.item(), loss_f.item())
            )
        return loss
        
    def resample_space(self):
        '''
        空间重采样:
        给训练集加上200个self.X_f_test中loss_f较大的点
        '''
        self.dnn.eval()
        f_pred = self.net_f(self.x_test, self.t_test)
        f_pred = f_pred.detach().cpu().numpy()
        index = np.argsort(f_pred.flatten())
        index = index[::-1]
        num = int(0.3 * len(index))
        num = 200
        index = index[:num]
        resample = self.X_f_test[index]
        self.X_f = np.vstack([self.X_f_retain, resample])  
        
        # 重设优化器?
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)
        self.optimizer_LBFGS = torch.optim.LBFGS(self.dnn.parameters(),
                                           max_iter=200,
                                           tolerance_grad=1.e-8,
                                           tolerance_change=1.e-12,
                                           history_size=50)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_Adam,
                                   step_size=100,
                                   gamma=0.65)
        
    def train(self):
        batch_size = 30
        best_loss = 1.e10
        for i in range(self.epoch_adam):
            print("------the {} trun train------".format(i + 1))
            self.dnn.train()
            for j in self.data_iter(batch_size, self.X_f):
                x_j = j[:, 0:1]
                t_j = j[:, 1:2]
                loss = self.loss_func_batch(x_j, t_j)
                self.optimizer_Adam.step()
            if (i + 1) % 20 == 0:
                is_best = loss < best_loss
                best_loss = loss if is_best else best_loss
                if is_best:
                    torch.save(self, "./model_adam.pth")
            self.lr_scheduler.step()
        best_loss = 1.e10
        for i in range(self.epoch_lbfgs):
            loss = self.train_LBFGS()
            is_best = loss < best_loss
            best_loss = loss if is_best else best_loss
            if is_best:
                torch.save(self, "./model_adam.pth")
            if loss < 1e-5:
                break
    
    def train_LBFGS(self):
        self.dnn.train()
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()
            loss = self.loss_func()
            if loss.requires_grad:
                loss.backward()
            return loss
        self.optimizer_LBFGS.step(closure)
        loss = closure()
        return loss.item()
    
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


# Configurations
N_u = 512
N_f = 2000
N_b = 200
layers = [2, 128, 128, 128, 128, 1]

data = scipy.io.loadmat('/home/stu1/liuyun/dongwenpang/NS_32/AC_minibatch_test1/AC.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]

Exact = np.real(data['uu']).T

X, T = np.meshgrid(x, t)

# X_star是一个二维数组，第一列表示空间坐标，第二列表示时间坐标
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# u_star是一个一维数组，表示X_star 时空坐标上的解
u_star = Exact.flatten()[:, None]

# 区域界限
lb = X_star.min(0)  # 最小点(-1, 0)
ub = X_star.max(0)  # 最大点(1, 0.99)

# xx1表示初始的点的时空坐标，第一列表示空间坐标，第二列表示时间坐标0
x_init = np.arange(-1, 1 + 1 / 800, 1 / 800)[:, None]
T_init = np.zeros_like(x_init)
xx1 = np.hstack((x_init, T_init))
X_u_train = np.vstack([xx1])

# 初始点处的函数值
uu1 = xx1[:, 0:1] ** 2 * np.cos(np.pi * xx1[:, 0:1])

# X[:, 0:1]:左边界,全是-1. T[:, 0:1]:0~1的一个时间剖分
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
X_b_l = np.vstack([xx2])
uu2 = Exact[:, 0:1]  # -1这个点在所有时刻上的函数值


xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]
X_b_r = np.vstack([xx3])   # 1这个点在所有时刻上的函数值
 
# 内部采样点，采样空间在(-1, 0)和(1, 0.99)之间
X_f_train = lb + (ub - lb) * lhs(2, N_f)  
X_f_train = np.vstack((X_f_train, xx1, xx2, xx3))  # N_f个采样点 + 所有初边值点
u_train = np.vstack([uu1])

# 初值点里面随机选择100个点
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]

# 初值点上对应的函数值
u_train = u_train[idx, :]

# 边值点上对应的函数值
idxb = np.random.choice(X_b_l.shape[0], N_b, replace=False)
X_b_l = X_b_l[idxb]
X_b_r = X_b_r[idxb]

# 采样大量的测试点
X_f_test = lb + (ub - lb) * lhs(2, 5 * N_f)


model = PhysicsInformedNN(X_u_train, u_train, X_b_l, X_b_r, X_f_train, layers, lb, ub,  X_f_test)

# 训练
for i in range(10):
    model.train()
    model.resample_space()
    if i == 2:
        torch.save(model, "resample_space/model_resample_2.pth")
    elif i == 5:
        torch.save(model, "resample_space/model_resample_5.pth")
    elif i == 8:
        torch.save(model, "resample_space/model_resample_8.pth")

torch.save(model, "resample_space/model_resample.pth")

# 预测
u_pred, f_pred = model.predict(X_star)
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


# 画图
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (u_train.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[100] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[150] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)
plt.savefig("resample_space/predict00.jpg")

####### Error Heat ##################
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(Error.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (u_train.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
    alpha=1.0
)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[100] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[150] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('Error in u(x,t)', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)
plt.savefig("resample_space/predict01.jpg")



####### Row 1: u(t,x) slices ##################

""" The aesthetic setting has changed. """

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 4)
gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0$', fontsize=15)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize=15)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[150, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[150, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.75$', fontsize=15)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 3])
ax.plot(x, Exact[200, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[200, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 1$', fontsize=15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.savefig("resample_space/predict02.jpg")
