# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:28:02 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

import Error
import numpy as np

d=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\CV.txt',delimiter='\t')
d1 =d[:,0]
d2 =d[:,1]
mae=Error.get_mae(d1,d2)
rmse=Error.get_rmse(d1,d2)
r2 = Error.get_r2(d1,d2)

print('平均绝对误差:{:.3f}'.format(mae))
print('均方根误差:{:.3f}'.format(rmse))
print('r2:{:.3f}'.format(r2))