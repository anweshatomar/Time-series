import numpy as np
import pandas as pd
import warnings
import scipy
import code_collection
import statsmodels.api as sm
from matplotlib import style
import seaborn as sns
from scipy.stats import chisquare
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
#style.use('ggplot')
np.set_printoptions(suppress=True)
data=pd.read_csv("T1.csv",header=0)
#selecting only the first 5000 data points
data=data[:5000]
data['Date/Time']=pd.to_datetime(data['Date/Time'])
date_check=data['Date/Time']
print("Start date:",date_check[0])
print("End date:",date_check.iloc[-1])
#split data
y_train=data[:4000]
y_test=data[4000:]
teta=[0.99618301,0.00491499,0.05504857,0.05561321,0.07856018,0.05308865,0.10017572,0.04026626]
#one step
y_t=y_train['LV ActivePower (kW)']
y_tt=y_test['LV ActivePower (kW)']
y_hat_t_1 = []
for i in range(0,len(y_t)):
    if i==0:
        y_hat_t_1.append(y_t[i]*teta[0] +teta[2]* y_t[i])
    elif i==1:
        y_hat_t_1.append(y_t[i]*teta[0] +teta[1]* y_t[i-1] + teta[2]*(y_t[i] - y_hat_t_1[i - 1] ) + teta[3]*(y_t[i - 1]))
    else:
        y_hat_t_1.append( y_t[i]*teta[0] +teta[1]* y_t[i-1] + teta[2]*(y_t[i] - y_hat_t_1[i - 1] ) + teta[3]*(y_t[i - 1] - y_hat_t_1[i-1])+ teta[4]*(y_t[i - 1]- y_hat_t_1[i-1])+teta[5]*(y_t[i - 1]- y_hat_t_1[i-1])+teta[6]*(y_t[i - 1]- y_hat_t_1[i-1])+teta[7]*(y_t[i - 1]- y_hat_t_1[i-1]))
#forecast function
y_hat_t_h = []
for h in range(0,len(y_tt)):
    if h==0:
        y_hat_t_h.append(y_t.iloc[-1]*teta[0] +teta[1]*y_t.iloc[-2]+ teta[2]*(y_t.iloc[-1] - y_hat_t_1[-2]) + teta[3]*(y_t.iloc[-2]-y_hat_t_1[-3]))
    elif h==1:
         y_hat_t_h.append(y_hat_t_h[h-1]*teta[0] + teta[1]*y_t.iloc[-1] + teta[3]*(y_t.iloc[-1] - y_hat_t_1[-2]))

    else:
        y_hat_t_h.append(y_hat_t_h[h-1]*teta[0] +teta[1]*y_hat_t_h[h-2] )
#errors,mse,rmse,acf,q value
error,mse=[],[]
for i,j in zip(y_test['LV ActivePower (kW)'],y_hat_t_h):
	error.append(i-j)
	mse.append((i-j)**2)
MSE=np.mean(mse)
rmse=np.sqrt(MSE)
acf=code_collection.calc_acf(error,120)
code_collection.plt_acf(acf,"ACF of ARMA H step",120)
q_val=code_collection.q_val(y_test['LV ActivePower (kW)'],acf)
plt.title("Manual Calcs- ARMA Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train['Date/Time'],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test['Date/Time'],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test['Date/Time'],y_hat_t_h,'r--')
plt.show()
print("MSE:",MSE)
print("RMSE",rmse)
print("Q value", q_val)