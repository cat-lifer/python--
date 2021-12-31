# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:36:58 2020

@author: hhy
"""

#### 配合 ‘一定要挑出来吖’ 使用，挑出来之后，使用最佳模型的参数进行测试
#### 数据集随机划分100次，预测100次，求平均R^2

#导入算法包
from sklearn.linear_model import Ridge,Lasso  #线性
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.svm import SVR  # 支持向量机
from sklearn.ensemble import RandomForestRegressor  #随机森林
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升回归
from sklearn.neural_network import MLPRegressor    #神经网络

#导入工具包
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import  train_test_split #随机划分


######################定义误差######################
def get_mse(records_real, records_predict):
        """
        均方误差 估计值与真值 偏差
        """
        if len(records_real) == len(records_predict):
            return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        else:
            return None
def get_rmse(records_real, records_predict):
        """
        均方根误差：是均方误差的算术平方根
        """
        mse = get_mse(records_real, records_predict)
        if mse:
            return math.sqrt(mse)
        else:
            return None
def get_mre(records_real, records_predict):
        """
        平均相对误差
        """
        if len(records_real) == len(records_predict):
            return sum([abs(x - y) for x, y in zip(records_real, records_predict)]/(y_test)) / len(records_real)
        else:
            return None


######################预留矩阵
#Score
RMSE = np.zeros(shape=(100,1))
MRE = np.zeros(shape=(100,1))
train_RR = np.zeros(shape=(100,1))
test_RR = np.zeros(shape=(100,1))
#feature_rank=np.zeros(shape=(12,100))


#加载数据
data=np.loadtxt(r'C:\Users\DELL\Desktop\代码集\数据库\V-453.txt',delimiter='\t') #读入txt数据，以tab为分界
for k in range(100):
    
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
    #将所有数据分为两类，80%作训练集，20%作预测集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=None)
    #数据预处理 ，SVR 和 MLP 需要
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #################根据最优参数建模,采用哪种算法，就从下面复制过来 ######################
    gbr = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=38)
    gbr_train= gbr.fit(X_train,y_train)
    pre_test= gbr.fit(X_train,y_train).predict(X_test)
    
    ##score
    rmse = get_rmse(y_test, pre_test)
    mre = get_mre(y_test, pre_test)
    RMSE[k,:] = rmse
    MRE[k,:] = mre
#    feature_rank[:,k] = importances
    train_RR[k,:]= gbr.score(X_train, y_train)
    test_RR[k,:] = gbr.score(X_test, y_test)

##结果拼接 RMSE MRE train_RR test_RR
Final_results = np.concatenate((RMSE,MRE,train_RR,test_RR),axis=1)
Final_results = pd.DataFrame(Final_results,
                             columns=['RMSE', 'MRE', 'train_RR','test_RR'])  
    

"""算法选择
    ##Ridge
    ridge = Ridge(alpha=0.1,max_iter=100)
    ridge_train = ridge.fit(X_train)
    pre_test =ridge.fit(X_train).predict(X_test)
    ##Lasso
    lasso = Lasso(alpha=0.1,max_iter=100)
    lasso_train = lasso.fit(X_train)
    pre_test =lasso.fit(X_train).predict(X_test)
    #KNN
    knn = KNeighborsRegressor(n_neighbors=4)
    knn_train=knn.fit(X_train,y_train)
    pre_test = knn.fit(X_train, y_train).predict(X_test)
    ##RFR
    rfr = RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=38)
    rfr_train = rfr.fit(X_train,y_train)
    pre_test = rfr.fit(X_train,y_train).predict(X_test)
    importances = rfr.feature_importances_    
    ##GBR
    gbr = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=38)
    gbr_train= gbr.fit(X_train,y_train)
    pre_test= gbr.fit(X_train,y_train).predict(X_test)
    importances = gbr.feature_importances_    
    ##SVR
    svr = SVR(svr__kernel='rbf',svr__C=0.1)
    svr_train[:,k] = svr.fit(X_train_scaled,y_train)
    pre_test[:,k] = svr.fit(X_train_scaled,y_train).predict(X_test_scaled)
    ##MLP
    mlp = MLPRegressor(random_state=38,max_iter=5000,activation ='tanh',
                       alpha=1,hidden_layer_sizes=(6,),solver='lbfgs')
    mlp_train[:,k] = mlp.fit(X_train_scaled,y_train)
    pre_test[:,k] = mlp.fit(X_train_scaled,y_train).predict(X_test_scaled)
"""