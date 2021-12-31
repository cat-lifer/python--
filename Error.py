# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:21:31 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""


# d=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\CV.txt',delimiter='\t')
# d1 =d[:,0]
# d2 =d[:,1]
# get_mae(d1,d2)
# get_rmse(d1,d2)
# get_r2(d1,d2)

### 计算模型评价参数
import math
import numpy as np
from scipy.stats import pearsonr

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


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_PCC(records_real, records_predict):
    '''
    真实值与预测值的相关性
    '''
    records_real = np.squeeze(records_real)
    records_predict = np.squeeze(records_predict)
    P_CC = pearsonr(records_real,records_predict)
    return P_CC[0]

def get_r2(records_real, records_predict):
    SStot=np.sum((records_real-np.mean(records_real))**2)
    SSres=np.sum((records_real-records_predict)**2)
    r2=1-SSres/SStot
    return r2

