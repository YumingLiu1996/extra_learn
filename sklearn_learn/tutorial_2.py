'''
本节用来实验fit, transform以及fit_transform
fit是用来对模型进行训练
transform是用来对将数据进行变换
fit_transform既对模型进行了训练，又对数据进行了变换

preprocessing中的两个函数
scale: 是将数据转换成均值为0，标准差为1的数据
StandardScaler函数会去学习数据的均值以及方差，得到的数据将用在测试数据上，对测试数据进行变换
'''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
x_train = np.array([[1,2,3],[2,3,4],[3,4,5]])
x_test = np.array([[1,2,3],[2,3,4],[3,4,5]])
sc = StandardScaler()


x_scale = scale(x_train)
print(x_scale.std(axis=0))


sc.fit(x_train)
x_train1 = sc.transform(x_train)
sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(sc.var_)
print(sc.scale_)
print(x_train1)
print(x_train)
print(x_test)
