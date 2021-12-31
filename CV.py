# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:58:52 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge,Lasso  #线性
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.svm import SVR  # 支持向量机
from sklearn.ensemble import RandomForestRegressor  #随机森林
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升回归
from sklearn.neural_network import MLPRegressor    #神经网络
from sklearn.gaussian_process import GaussianProcessRegressor   #高斯过程回归
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
"""
模型模板
model = Ridge(alpha = .5,max_iter=1000)
model = Lasso(alpha = .5,max_iter=1000)
model = KNeighborsRegressor(n_neighbors=6)
model = SVR(gamma='auto_deprecated',kernel='rbf',C=1.0,degree=3)
model = RandomForestRegressor(n_estimators=100,max_depth=5)
model = GradientBoostingRegressor(n_estimators=100,max_depth=5)
model = MLPRegressor(random_state=0,max_iter=50000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(8,),solver='lbfgs')
model =GaussianProcessRegressor(kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10)),
                                alpha=0.01,n_restarts_optimizer=10)
"""
data=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\数据库\153life.txt',
                delimiter='\t')
X = data[:,:13]
scaler = StandardScaler()
scaler.fit(X)
X1 = scaler.transform(X)
data=np.concatenate((X1,data[:,-1].reshape(153,1)),axis=1) #不归一化的时候将X1改为X

indices = np.arange(153)
k_fold = KFold(n_splits=10, shuffle=True)
train_test_set = k_fold.split(indices)

sheet =['cv1','cv2','cv3','cv4','cv5','cv6','cv7','cv8','cv9','cv10']
n=0
end = "C:/Users/Uaena_HY/Desktop/结果汇总.xlsx" 
writer = pd.ExcelWriter(end)
for (train_set, test_set) in train_test_set:
    n=n+1
    train = data[train_set,:]
    test = data[test_set,:]
   
    x_train = train[:,:13]
    y_train = train[:,-1]
    x_test = test[:,:13]
    y_test = test[:,-1]
        
    pre=np.zeros(shape=(len(y_test),10))
    for i in range(10):
        model = MLPRegressor(random_state=0,max_iter=2000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(10,),solver='lbfgs')


        model.fit(x_train,y_train)
        pre[:,i] = model.predict(x_test)
    pre_test=np.concatenate((x_test,y_test.reshape(len(y_test),1),
                              (pre.mean(axis=1)).reshape(len(y_test),1)),axis=1)
    pre_test=pd.DataFrame(pre_test)
    
    pre_test.to_excel(writer,sheet_name=sheet[n-1],index=False)
    
    if n<11:
        continue
writer.save()
writer.close()
   