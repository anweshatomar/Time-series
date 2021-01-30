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
target=data['LV ActivePower (kW)']
data['LV ActivePower (kW)']= target.diff().dropna(axis=0)
y_train=target[:4000]
y_test=target[4000:]
#x_train, x_test,y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=42, shuffle=False)
#EDA-data conversion
data['LV ActivePower (kW)']=data['LV ActivePower (kW)'][1:]
sm.graphics.tsa.plot_acf(data['LV ActivePower (kW)'], lags=40)
plt.show()
data_acf=code_collection.calc_acf(data['LV ActivePower (kW)'],10)
code_collection.plt_acf(data_acf,"LV ActivePower (kW) for 20 lags",10)

gpac_data=code_collection.cal_gpac(data_acf,8,8)
#GPAC
gpac_matrix=np.asmatrix(gpac_data)
print("GPAC Values:")
column_labels=[]
row_labels=[]
for i in range(1,len(gpac_matrix)+1):
    column_labels.append(i)
    row_labels.append(i-1)
df = pd.DataFrame(gpac_data, columns=column_labels, index=row_labels)
print(df)
