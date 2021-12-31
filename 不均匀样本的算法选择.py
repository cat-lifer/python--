# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:24:42 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""
'''
################# 针对数据分布不均匀数据库的算法选择 ####################
## 先用一般算法选择模块，确定模型参数后，根据泛化能力最大原则进一步选择算法
## 原理：评价模型对测试集中偏离训练集样本较大的数据的预测精度
'''
############################################################################################
#导入算法包  10种回归算法
from sklearn.linear_model import Ridge,Lasso  #线性
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.svm import SVR  # 支持向量机
from sklearn.ensemble import RandomForestRegressor  #随机森林
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升决策树
from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPRegressor    #神经网络

#导入工具包
import numpy as np
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import train_test_split

#导入画图包
import matplotlib.pyplot as plt 

############################################################################################
#加载数据
data=np.loadtxt(r'C:\Users\DELL\Desktop\代码集\数据库\V-403.txt',delimiter='\t') #读入txt数据，以tab为分界
#数据集打乱
data=np.random.permutation(data)
x = []
y = []
for line in range(len(data)):
    x.append(data[line][:8])
    y.append(data[line][-1])
X = np.array(x)
y = np.array(y)

scaler = StandardScaler()
scaler.fit(X)
X1 = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=38)
X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y,test_size=0.1,random_state=38)

############################################################################################
### 欧氏距离
import math
def euclidean(x, y):
 d = 0.
 for xi, yi in zip(x, y):
     d += (xi-yi)**2
 return math.sqrt(d)

### 求每一个test样本 与 train样本 的欧氏距离   最后取最小值
distance = np.zeros(shape=(len(X_train),len(X_test)))
for i in range(len(X_test)):
    for k in range(len(X_train)):
        distance[k,i] = euclidean(X_test[i][:],X_train[k][:])

meandistance = np.min(distance,0)   #按列    
############################################################################################

### Model
Ridge = Ridge(alpha=1.0,max_iter=100)
Lasso = Lasso(alpha=0.01,max_iter=100)
KNN = KNeighborsRegressor(n_neighbors=3)
RFR = RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=38)
GBRT = GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=38)
XGBoost = XGBRegressor(learning_rate=0.1,n_estimators=200)
SVR = SVR(kernel='poly',degree=3,C=1,gamma='auto')
MLP = MLPRegressor(random_state=38,max_iter=5000,activation='relu',solver='lbfgs',
                   hidden_layer_sizes=(8,),alpha=0.01)

### fit  Predict
Ridge_test = Ridge.fit(X_train, y_train).predict(X_test)
Lasso_test = Lasso.fit(X_train, y_train).predict(X_test)
KNN_test = KNN.fit(X_train, y_train).predict(X_test)
RFR_test = RFR.fit(X_train, y_train).predict(X_test)
GBRT_test = GBRT.fit(X_train, y_train).predict(X_test)
XGBoost_test = XGBoost.fit(X_train, y_train).predict(X_test)
SVR_test = SVR.fit(X1_train, y1_train).predict(X1_test)
MLP_test = MLP.fit(X1_train, y1_train).predict(X1_test)

### 评分
Ridge_score = Ridge.score(X_test, y_test)
Lasso_score = Lasso.score(X_test, y_test)
KNN_score = KNN.score(X_test, y_test)
RFR_score = RFR.score(X_test, y_test)
GBRT_score = GBRT.score(X_test, y_test)
XGBoost_score = XGBoost.score(X_test, y_test)
SVR_score = SVR.score(X1_test, y1_test)
MLP_score = MLP.score(X1_test, y1_test)

Score = np.hstack((Ridge_score,Lasso_score,KNN_score,RFR_score,GBRT_score,
                   XGBoost_score,SVR_score,MLP_score))

### RE
def get_re(records_real, records_predict):                
        if len(records_real) == len(records_predict):
            return [abs(x - y) for x, y in zip(records_real, records_predict)]/(y_test)
        else:
            return None
 
Ridge_testerror =  get_re(y_test,Ridge_test)
Lasso_testerror =  get_re(y_test,Lasso_test)   
KNN_testerror =  get_re(y_test,KNN_test)
RFR_testerror =  get_re(y_test,RFR_test)
GBRT_testerror =  get_re(y_test,GBRT_test)
XGBoost_testerror =  get_re(y_test,XGBoost_test)  
SVR_testerror =  get_re(y_test,SVR_test)
MLP_testerror =  get_re(y_test,MLP_test) 

############################################################################################
### 结果整理 [distance Ridge Lasso KNN RFR GBRT XGBoost SVR MLP]
L =len(X_test)
end = np.hstack((meandistance.reshape(L,1),Ridge_testerror.reshape(L,1),
                 Lasso_testerror.reshape(L,1),KNN_testerror.reshape(L,1),
                 RFR_testerror.reshape(L,1),GBRT_testerror.reshape(L,1),
                 XGBoost_testerror.reshape(L,1),SVR_testerror.reshape(L,1),
                 MLP_testerror.reshape(L,1)))
end = end[end[:,0].argsort()] #按distance的顺序排序
############################################################################################
### 可视化
miny = np.min(np.min(end[:,1:8],axis=0))
maxy = np.max(np.max(end[:,1:8],axis=0))
xx = end[:,0]
namelist = ['Ridge', 'Lasso', 'KNN', 'RFR',' GBRT', 'XGBoost', 'SVR', 'MLP']
for p,name in enumerate(namelist):
    plt.subplot(3,3,p+1) 
    plt.bar(xx, end[:,p+1], 0.4, color="g")
    plt.title(name,fontsize=12) #设置标题
    plt.ylim(miny,maxy)
    plt.xlabel('Min(distance) from trainset',fontsize=10)
    plt.ylabel('Relative error',fontsize=10)
    plt.subplots_adjust(wspace=0.5,
                    hspace=0.5,
                    left=0.125,
                    right=0.9,
                    top=0.9,
                    bottom=0.1)
plt.subplot(3,3,9)
plt.bar(range(len(Score)), Score,color='b',tick_label=namelist)
plt.xticks(rotation=90)
plt.ylabel('Test R^2',fontsize=10)
plt.ylim(0,1)
plt.show()