import numpy as np

gp = np.array([0.8611363116,-0.8611363116,0.3399810436,-0.3399810436])[None, :]
gc = np.array([0.3478548451,0.3478548451,0.6521451549,0.6521451549])[None, :]

# 在区间[-1, 1]内生成100个高斯积分点和对应的高斯权重

l = np.linspace(-1, 1, 26)[:, None]
l = np.hstack([l[:-1], l[1:]])
c = (l[:, 1] - l[:, 0])/2
d = (l[:, 1] + l[:, 0])/2
c = c[:, None]
d = d[:, None]

n_p  = c * gp +d 
n_c = c * gc
n_p = n_p.reshape(1, 100)
n_c = n_c.reshape(1, 100)


# 验证: 在(-1, 1)对cos(\pi x) **2 做积分
q = np.cos(np.pi * n_p) ** 2 * n_c
q = np.sum(q)
print(q)  #输出 0.9999999999999998
