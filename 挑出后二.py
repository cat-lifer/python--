# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:31:39 2020

@author: hongyong Han

To: Do or do not. There is no try.

"""

#### 配合 ‘一定要挑出来吖’ 使用，挑出来之后，使用最佳模型的参数进行测试
#### 目的：获得预测值--真实值数据和曲线，，划分一次，预测100次取平均

#导入算法包
from sklearn.linear_model import Ridge,Lasso  #线性
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.svm import SVR  # 支持向量机
from sklearn.ensemble import RandomForestRegressor  #随机森林
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升回归
from sklearn.neural_network import MLPRegressor    #神经网络

#导入工具包
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import  train_test_split #随机划分


data= np.loadtxt(r'C:\Users\DELL\Desktop\代码集\数据库\V-453.txt',delimiter='\t')
#数据集打乱
data=np.random.permutation(data)
#分输入输出数据
x = []
y = []
for line in range(453):
    x.append(data[line][:9])
    y.append(data[line][-1])
X = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=None)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

num1=len(y_train)
num2=len(y_test)
pre_y_train=np.zeros((num1,100),dtype=float)  #训练集的预测值
pre_y_test = np.zeros((num2,100),dtype=float)   #测试集的预测值

for k in range(100):   # 只有以下两行需要改
    
    knn = KNeighborsRegressor(n_neighbors=3)
    model=knn.fit(X_train,y_train)
    
    pre_train=model.predict(X_train)
    pre_test = model.predict(X_test)
    pre_y_train[:,k]=pre_train
    pre_y_test[:,k]=pre_test
    
train_predicted = np.mean(pre_y_train,axis=1)
train_real = y_train
test_predicted = np.mean(pre_y_test,axis=1)
test_real = y_test

Train = np.concatenate((train_real.reshape(num1,1),train_predicted.reshape(num1,1)),axis=1)
Test = np.concatenate((test_real.reshape(num2,1),test_predicted.reshape(num2,1)),axis=1)
## 只需要看这个表就够了
Final_Train =pd.DataFrame(Train,columns=['Train Real','Train Predict'])
Final_Test =pd.DataFrame(Test,columns=['Test Real','Test Predict'])
## 可视化
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(train_real,train_predicted,color='red',label='Train')
plt.scatter(test_real,test_predicted,color='blue',label='Test')
plt.title('Gamma Prime AreaFraction',fontsize=18) #设置标题，每次更改
plt.xlabel('Real',fontsize=18)
plt.ylabel('Predicted',fontsize=18)
plt.yticks(size=18)
plt.xticks(size=18)
plt.legend() #显示图例
plt.show()

"""算法选择
    ##Ridge
    ridge = Ridge(alpha=0.1,max_iter=100)
    model = ridge.fit(X_train)
    ##Lasso
    lasso = Lasso(alpha=0.1,max_iter=100)
    model = lasso.fit(X_train)
    #KNN
    knn = KNeighborsRegressor(n_neighbors=4)
    model=knn.fit(X_train,y_train)
    ##RFR
    rfr = RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=38)
    model = rfr.fit(X_train,y_train)
    importances = rfr.feature_importances_    
    ##GBR
    gbr = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=38)
    model= gbr.fit(X_train,y_train)
    importances = gbr.feature_importances_    
    ##SVR
    svr = SVR(svr__kernel='rbf',svr__C=0.1)
    model = svr.fit(X_train_scaled,y_train)
    pre_train=model.predict(X_train_scaled)
    pre_test = model.predict(X_test_scaled)
    ##MLP
    mlp = MLPRegressor(random_state=38,max_iter=5000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(9,),solver='lbfgs')
    model = mlp.fit(X_train_scaled,y_train)
    
    pre_train=model.predict(X_train_scaled)
    pre_test = model.predict(X_test_scaled)
"""