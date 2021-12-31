# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:09:48 2019

@author: hhy_The Grand Design
"""
import time  #为了计算程序运行的时间
start =time.perf_counter()
import warnings
warnings.filterwarnings("ignore")  #为了美观不显示警告

#导入算法包  10种回归算法
from sklearn.linear_model import Ridge,Lasso  #线性
from sklearn.neighbors import KNeighborsRegressor  #k-NN
from sklearn.svm import SVR  # 支持向量机
from sklearn.ensemble import RandomForestRegressor  #随机森林
from sklearn.ensemble import GradientBoostingRegressor  #梯度提升决策树
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor    #Bagging 元估计  (装袋)
from sklearn.ensemble import AdaBoostRegressor   #自适应 Boosting
from sklearn.neural_network import MLPRegressor    #神经网络
from sklearn.gaussian_process import GaussianProcessRegressor   #高斯过程回归
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#导入工具包
import numpy as np
from sklearn.preprocessing import StandardScaler   #预处理 均值为0 方差为1
from sklearn.model_selection import cross_val_score  #交叉验证
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV   #网格搜索，将交叉验证和网格搜索封装在一起
from sklearn.pipeline import Pipeline   #导入管道模型
from sklearn.feature_selection import SelectFromModel   #特征选择，可以用管道模型统一

#导入画图包
import matplotlib.pyplot as plt 


#加载数据
data=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\75.txt',delimiter='\t') #读入txt数据，以tab为分界
#数据集打乱
data=np.random.permutation(data)
#分输入输出数据
x = []
y = []
for line in range(75):
    x.append(data[line][:16])
    y.append(data[line][-1])
X = np.array(x)
y = np.array(y)

scaler = StandardScaler()
scaler.fit(X)
X1 = scaler.transform(X)
###模型粗选 Ridge Lasso K-NN RFR GBRT SVR MLP    ##Bagging  AdaBoost 
#cv = LeaveOneOut() # "挨个儿试试"  数据量多时不要用，耗时
cv = 5 
## Ridge 
params_1 = {'alpha':[0.01,0.1,1.0,10.0],'max_iter':[100,1000,5000]}   #将需要遍历的参数定义为字典
grid_1 = GridSearchCV (Ridge(),params_1,cv=cv) #定义网格搜索中使用的模型和参数
grid_1.fit(X1,y)

##Lasso
params_2 = {'alpha':[0.01,0.1,1.0,10.0],'max_iter':[100,1000,5000]}
grid_2 = GridSearchCV (Lasso(),params_2,cv=cv)
grid_2.fit(X1,y)

##K-NN
params_3 = {'n_neighbors':[2,3,4,5,6,7,8,9,10]} 
grid_3 = GridSearchCV (KNeighborsRegressor(),params_3,cv=cv)
grid_3.fit(X1,y)

##RFR
params_4 = {'n_estimators':[100,200,500,1000],'max_depth':[3,4,5,6,7,8,9,10]}
grid_4 = GridSearchCV (RandomForestRegressor(random_state=38),params_4,cv=cv)
grid_4.fit(X,y)

##GBRT
params_5 = {'n_estimators':[100,200,500,1000],'max_depth':[3,4,5,6,7,8,9,10]}
grid_5 = GridSearchCV (GradientBoostingRegressor(random_state=38),params_5,cv=cv)
grid_5.fit(X,y)

##XGBoost
para_grid = {'xgbregressor__learning_rate': [0.01,0.1,0.5],'xgbregressor__n_estimators':[100,200,500,1000]}
grid = GridSearchCV(XGBRegressor(), param_grid = para_grid,cv=5)
grid.fit(X,y)

##SVR
pipeline = Pipeline([('scaler',StandardScaler()),('svr',SVR())])
params_6 = [{'svr__kernel':['rbf'],'svr__C':[1,10,20,30,50,80,100], 'svr__gamma':[1e-7,1e-6,1e-5,1e-4,1e-2,0.0001]},
            {'svr__kernel':['poly'],'svr__C':[1,10,20,30,50,80,100],'svr__degree':[3,4,5]},
            {'svr__kernel':['linear'],'svr__C':[1,10,20,30,50,80,100]},
            {'svr__kernel':['sigmoid'],'svr__C':[1,10,20,30,50,80,100]}]
grid_6 = GridSearchCV (pipeline,params_6,cv=cv)
grid_6.fit(X,y)

##MLP   隐含层节点数设置为 输入特征数*0.75  左右
pipe = Pipeline([('scaler',StandardScaler()),('mlp',MLPRegressor(random_state=38,max_iter=5000))])
params_7 = [{'mlp__activation':['logistic'],'mlp__alpha':[0.0001,0.001,0.01,1],'mlp__hidden_layer_sizes':[(9,),(10,),(11,)],'mlp__solver':['lbfgs','sgd','adam']},
            {'mlp__activation':['tanh'],'mlp__alpha':[0.0001,0.001,0.01,1],'mlp__hidden_layer_sizes':[(9,),(10,),(11,)],'mlp__solver':['lbfgs','sgd','adam']},
            {'mlp__activation':['relu'],'mlp__alpha':[0.0001,0.001,0.01,1],'mlp__hidden_layer_sizes':[(9,1),(10,),(11,)],'mlp__solver':['lbfgs','sgd','adam']}]
grid_7 = GridSearchCV (pipe,params_7,cv=cv)
grid_7.fit(X,y)

##GPR  高斯过程回归
kernel = C(0.1, (0.001,0.1))*RBF(0.5,(1e-4,10))
gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1)
gpr_score = cross_val_score(gpr,X1,y,cv=cv)

##打印结果   交叉验证的平均分的最高分及其对应的参数    分数指的是R的平方
print('\n')
print('Ridge 模型最高分:{:.3f}'.format(grid_1.best_score_))
print('Ridge 最优参数：{}'.format(grid_1.best_params_))
print('\n')
print('Lasso 模型最高分:{:.3f}'.format(grid_2.best_score_))
print('Lasso 最优参数：{}'.format(grid_2.best_params_))
print('\n')
print('k-NN 模型最高分:{:.3f}'.format(grid_3.best_score_))
print('k-NN 最优参数：{}'.format(grid_3.best_params_))
print('\n')
print('RFR 模型最高分:{:.3f}'.format(grid_4.best_score_))
print('RFR 最优参数：{}'.format(grid_4.best_params_))
print('\n')
print('GBRT 模型最高分:{:.3f}'.format(grid_5.best_score_))
print('GBRT 最优参数：{}'.format(grid_5.best_params_))
print('\n')
print('XGBoost 模型最高分:{:.3f}'.format(grid.best_score_))
print('XGBoost 最优参数：{}'.format(grid.best_params_))
print('\n')
print('SVR 模型最高分:{:.3f}'.format(grid_6.best_score_))
print('SVR 最优参数：{}'.format(grid_6.best_params_))
print('\n')
print('MLP 模型最高分:{:.3f}'.format(grid_7.best_score_))
print('MLP 最优参数：{}'.format(grid_7.best_params_))
print('\n')
print('GPR 交叉验证平均分:{:.3f}'.format(gpr_score.mean()))
print('\n')

##画柱状图
name_list = ['Ridge','Lasso','kNN','RFR','GBRT','XGBoost','SVR','MLP','GPR']
num_list = [grid_1.best_score_,grid_2.best_score_,grid_3.best_score_,
            grid_4.best_score_,grid_5.best_score_,grid.best_score_,
            grid_6.best_score_,grid_7.best_score_,gpr_score.mean()]
plt.bar(range(len(num_list)), num_list,tick_label=name_list,label='R^2')
plt.show()

end = time.perf_counter()
print('Running time: %s Minutes'%((end-start)/60))
print('\n')

'''
#BaggingRegressor  knn估计器  #第一遍运行先舍掉，得到最好的算法作为估计器
params_8 = {'n_estimators':[50,100,200,500],'max_samples':[0.8,0.9,1.0],'max_features':[0.8,0.9,1.0]}
grid_8 = GridSearchCV (BaggingRegressor(KNeighborsRegressor(n_neighbors=5),random_state=38),params_8)
grid_8.fit(X,y)

#AdaBoost    knn估计器  #第一遍运行先舍掉，得到最好的算法作为估计器
params_9 = {'n_estimators':[100,200,500,1000]}
grid_9 = GridSearchCV (AdaBoostRegressor(KNeighborsRegressor(n_neighbors=5),random_state=38),params_9)
grid_9.fit(X,y)
print('Bagging 模型最高分:{:.3f}'.format(grid_8.best_score_))
print('Bagging 最优参数：{}'.format(grid_8.best_params_))
print('\n')
print('AdaBoost 模型最高分:{:.3f}'.format(grid_9.best_score_))
print('AdaBoost 最优参数：{}'.format(grid_9.best_params_))
print('\n')

'''