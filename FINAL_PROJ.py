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
warnings.filterwarnings("ignore")
style.use('ggplot')
np.set_printoptions(suppress=True)
data=pd.read_csv("T1.csv",header=0)
#selecting only the first 5000 data points
data=data[:5000]
date_check=data['Date/Time']
print("Start date:",date_check[0])
print("End date:",date_check.iloc[-1])
#data['LV ActivePower (kW)']=data['LV ActivePower (kW)']-np.mean('LV ActivePower (kW)')
#data=data-np.mean(data)
data1=data[:144]
#EDA-data conversion
data['Date/Time']=pd.to_datetime(data['Date/Time'])
print(data.head(0))
print(data.dtypes)
print(len(data))
plt.plot(data['Date/Time'],data['LV ActivePower (kW)'])
plt.xlabel('Date and Time')
plt.ylabel('LV ActivePower (kW)')
plt.show()
#ADF test on dependent variable
X = data['LV ActivePower (kW)']
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
data_acf=code_collection.calc_acf(data['LV ActivePower (kW)'],20)
code_collection.plt_acf(data_acf,"LV ActivePower (kW) for 20 lags",20)
data_acf_80=code_collection.calc_acf(data['LV ActivePower (kW)'],80)
code_collection.plt_acf(data_acf_80,"LV ActivePower (kW) for 80 lags",80)
data_acf_120=code_collection.calc_acf(data['LV ActivePower (kW)'],120)
code_collection.plt_acf(data_acf_120,"LV ActivePower (kW) for 120 lags",120)

#gpac_data=code_collection.GPAC(data_acf,8,8)
gpac_data=code_collection.cal_gpac(data_acf,8,8)
print("Correlation between: LV ActivePower (kW)-Wind Speed (m/s)")
print(scipy.stats.pearsonr(data['LV ActivePower (kW)'],data['Wind Speed (m/s)'])[0])
print("Correlation between: LV ActivePower (kW)-Theoretical_Power_Curve (KWh)")
print(scipy.stats.pearsonr(data['LV ActivePower (kW)'],data['Theoretical_Power_Curve (KWh)'])[0])
print("Correlation between: LV ActivePower (kW)-Wind Direction (°)")
print(scipy.stats.pearsonr(data['LV ActivePower (kW)'],data['Wind Direction (°)'])[0])
#plot correlation matrix
corr = data.corr()
#plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr, vmin = -1, vmax = 1, annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
#de trended data
#decpose seasonality
lv_act_pow1 = data['LV ActivePower (kW)']
#additive
result1 = seasonal_decompose(lv_act_pow1, model="additive",period=240)
result1.plot()
plt.show()
de_seasonal=data['LV ActivePower (kW)']-result1.seasonal
de_trended=data['LV ActivePower (kW)']-result1.trend
#de trended data
plt.title("De Trended data plot")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(data['Date/Time'],data['LV ActivePower (kW)'],label="Original data")
plt.plot(data['Date/Time'],de_trended,label="After removing trend data")
plt.legend()
plt.show()
#seasonally adjusted data
plt.title("Removing Seasonality data plot")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(data['Date/Time'],data['LV ActivePower (kW)'],label="Original data")
plt.plot(data['Date/Time'],de_seasonal,label="After removing sesonal data")
plt.legend()
plt.show()
#stength
st=result1.seasonal
tt=result1.trend
rt=result1.resid
#calculating strength trend
vart1=np.var(rt)
vart2=np.var( tt+ rt)
ft_1=1-(vart1/vart2)
ft_0=0-(vart1/vart2)
ft=max(ft_1,ft_0)
print("The strength of trend for data set is:",ft)
# calculating strength seasonality
vars1 = np.var(rt)
vars2 = np.var(rt + st)
fs_1 = 1 - (vars1 / vars2)
fs_0 = 0 - (vars1 / vars2)
fs = max(fs_1, fs_0)
print("The strength of seasonality for data set is:", fs)
#backward regression
print("Using Wind Speed (m/s), Theoretical_Power_Curve (KWh),Wind Direction (°)")
elim1=['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)','Wind Direction (°)']
train1=data[elim1]
test1=data['LV ActivePower (kW)']
train1 = sm.add_constant(train1)
model1 = sm.OLS(test1, train1).fit()
print("Metrics:")
print("Adj R2", model1.rsquared_adj)
print("AIC", model1.aic)
print("BIC", model1.bic)
print("F statistic", model1.fvalue)
print("F p-value", model1.f_pvalue)
print("Using Wind Speed (m/s), Theoretical_Power_Curve (KWh)")
elim2=['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)']
train2=data[elim2]
test2=data['LV ActivePower (kW)']
train2 = sm.add_constant(train2)
model2 = sm.OLS(test2, train2).fit()
print("Metrics:")
print("Adj R2", model2.rsquared_adj)
print("AIC", model2.aic)
print("BIC", model2.bic)
print("F statistic", model2.fvalue)
print("F p-value", model2.f_pvalue)
print("Using Wind Speed (m/s)")
elim3=['Wind Speed (m/s)']
train3=data[elim2]
test3=data['LV ActivePower (kW)']
train3 = sm.add_constant(train3)
model3 = sm.OLS(test2, train3).fit()
print("Metrics:")
print("Adj R2", model3.rsquared_adj)
print("AIC", model3.aic)
print("BIC", model3.bic)
print("F statistic", model3.fvalue)
print("F p-value", model3.f_pvalue)
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
#split data
y_train=data[:4000]
y_test=data[4000:]
d_ml=data['Date/Time']
d_ml_train=d_ml[:4000]
d_ml_test=d_ml[4000:]
x=data.drop(columns=['LV ActivePower (kW)','Date/Time','Theoretical_Power_Curve (KWh)'])
y=data['LV ActivePower (kW)']
#multiple linear regression
x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, test_size=0.20, random_state=42, shuffle=False)
X = sm.add_constant(x_train1)
model = sm.OLS(y_train1,X).fit()
print("Multiple linear regression\n", model.summary())
#perform one step prediction
added_values = sm.add_constant(x_test1)
pred_ml=model.predict(added_values)
#plot data
plt.title("Multiple linear regression")
plt.xlabel("Date and Time")
plt.ylabel("LV Active Power")
plt.plot(d_ml_train,y_train1,label='Train data')
plt.plot(d_ml_test,y_test1, label='Test data')
plt.plot(d_ml_test,pred_ml,label='Predicted data')
plt.legend()
plt.show()
res_l=y_test1-pred_ml
#test for multiple linear regression
print("Metrics:")
print("Adj R2", model.rsquared_adj)
print("AIC", model.aic)
print("BIC", model.bic)
print("F statistic", model.fvalue)
print("F p-value", model.f_pvalue)
res_l=list(res_l)
acf_model_l=code_collection.calc_acf(res_l,lags=120)
code_collection.plt_acf(acf_model_l,"Plot ACF of Multilinear Linear Regression",120)
q_vv=code_collection.q_val(y_test1,acf_model_l)
print("Q value",q_vv)
print("Mean of residue",np.mean(res_l))
print("Variance of residue",np.var(res_l))
#y_train,y_test=train_test_split(data,shuffle=False, test_size=0.20)
y_test_drift=y_test[:-1]
len(y_test_drift)
#prediction_avg=pd.DataFrame({"Month": test["Month"], "#Passengers":avg})
#base models
y_train['LV ActivePower (kW)']=y_train['LV ActivePower (kW)']-np.mean(y_train['LV ActivePower (kW)'])
avg,av_er,av_mse=code_collection.calc_avg(y_train['LV ActivePower (kW)'],y_test['LV ActivePower (kW)'])
avg=avg+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- Average Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train["Date/Time"],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test["Date/Time"],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test["Date/Time"],avg,'r--')
plt.show()
naive,nav_er,nav_mse=code_collection.calc_naive(y_train['LV ActivePower (kW)'],y_test['LV ActivePower (kW)'])
naive=naive+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- Naive Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train["Date/Time"],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test["Date/Time"],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test["Date/Time"],naive,'r--')
plt.show()
drift,d_er,d_mse=code_collection.calc_drift(y_train['LV ActivePower (kW)'],y_test['LV ActivePower (kW)'])
drift=drift+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- Drift Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train["Date/Time"],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test["Date/Time"],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test_drift["Date/Time"],drift,'r--')
plt.show()
ses,s_er,s_mse=code_collection.calc_drift(y_train['LV ActivePower (kW)'],y_test['LV ActivePower (kW)'])
ses=ses+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- SES Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train["Date/Time"],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test["Date/Time"],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test_drift["Date/Time"],ses,'r--')
plt.show()
#printing base models only with test set
split_day=y_test["Date/Time"]
split_lv=y_test['LV ActivePower (kW)']
avg_day=avg
naive_day=naive
drift_day=drift
ses_day=ses
plt.title("Daily data- Naive Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],naive_day[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
plt.title("Daily data- Drift Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],drift_day[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
plt.title("Daily data- SES Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],ses_day[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
#arma process
arma_date=y_test['LV ActivePower (kW)']
#start_date=arma_date.loc[0]
#end_date=arma_date.loc[-1]
arma_error_1,arma_error_2, arma_error_3, arma_error_4=[],[],[],[]
arma_1=sm.tsa.ARMA(y_train['LV ActivePower (kW)'], order=(1,2)).fit(disp=False)
arma_pred_1=arma_1.forecast(len(y_test['LV ActivePower (kW)']))[0]
arma_pred_1=arma_pred_1+np.mean(y_train['LV ActivePower (kW)'])
for i,j in zip(y_test['LV ActivePower (kW)'],arma_pred_1):
	arma_error_1.append(i-j)
arma_2=sm.tsa.ARMA(y_train['LV ActivePower (kW)'], order=(1,7)).fit(disp=False)
arma_pred_2=arma_2.forecast(len(y_test['LV ActivePower (kW)']))[0]
arma_pred_2=arma_pred_2+np.mean(y_train['LV ActivePower (kW)'])
for i,j in zip(y_test['LV ActivePower (kW)'],arma_pred_2):
	arma_error_2.append(i-j)
arma_3=sm.tsa.ARMA(y_train['LV ActivePower (kW)'], order=(3,2)).fit(disp=False)
arma_pred_3=arma_3.forecast(len(y_test['LV ActivePower (kW)']))[0]
arma_pred_3=arma_pred_3+np.mean(y_train['LV ActivePower (kW)'])
for i,j in zip(y_test['LV ActivePower (kW)'],arma_pred_3):
	arma_error_3.append(i-j)
arma_4=sm.tsa.ARMA(y_train['LV ActivePower (kW)'], order=(3,7)).fit(disp=False)
arma_pred_4=arma_4.forecast(len(y_test['LV ActivePower (kW)']))[0]
arma_pred_4=arma_pred_4+np.mean(y_train['LV ActivePower (kW)'])
for i,j in zip(y_test['LV ActivePower (kW)'],arma_pred_4):
	arma_error_4.append(i-j)
ch1=chisquare(arma_error_1)[1]
ch2=chisquare(arma_error_2)[1]
ch3=chisquare(arma_error_3)[1]
ch4=chisquare(arma_error_4)[1]
r_a1=np.sqrt(np.mean(arma_error_1))
r_a2=np.sqrt(np.mean(arma_error_2))
r_a3=np.sqrt(np.mean(arma_error_3))
r_a4=np.sqrt(np.mean(arma_error_4))
print("Chi Sq test for ARMA (1,2)", ch1)
print("Chi Sq test for ARMA (1,7)", ch2)
print("Chi Sq test for ARMA (3,2)", ch3)
print("Chi Sq test for ARMA (3,7)", ch4)
#RMSE
print("RMSE for ARMA (1,2)", r_a1)
print("RMSE for ARMA (1,7)", r_a2)
print("RMSE for ARMA (3,2)", r_a3)
print("RMSE for ARMA (3,7)", r_a4)
print("FINAL ARMA ORDER:(1,7)")
arma=sm.tsa.ARMA(y_train['LV ActivePower (kW)'], order=(1,7)).fit(disp=False)
arma_pred=arma.forecast(len(y_test['LV ActivePower (kW)']))[0]
arma_pred=arma_pred+np.mean(y_train['LV ActivePower (kW)'])
#arma model details
print("ARMA model summary")
print(arma.summary())
print("ARMA confidence interval")
print(arma.conf_int())
plt.title("All data- ARMA Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train["Date/Time"],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test["Date/Time"],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test["Date/Time"],arma_pred,'r--')
plt.show()

plt.title("Daily data- ARMA Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],arma_pred[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
arma_error,mse_arma=[],[]
for i,j in zip(y_test['LV ActivePower (kW)'],arma_pred):
	arma_error.append(i-j)
	mse_arma.append((i-j)**2)
#Holt winter model
model2=ets.Holt(y_train['LV ActivePower (kW)'], initialization_method="estimated").fit()
holt_pred=model2.forecast(len(y_test['LV ActivePower (kW)']))
holt_pred=holt_pred+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- Holt Winter Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train['Date/Time'],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test['Date/Time'],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test['Date/Time'],holt_pred,'r--')
plt.xticks(rotation=90)
plt.show()
plt.title("Daily data- Holt Winter Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],holt_pred[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
holt_error,mse_holt=[],[]
for i,j in zip(y_test['LV ActivePower (kW)'],holt_pred):
	holt_error.append(i-j)
	mse_holt.append((i-j)**2)
#holt seasonal model
model_holt_seasonal=ets.ExponentialSmoothing(y_train['LV ActivePower (kW)'], initialization_method="estimated").fit()
holt_pred_s=model_holt_seasonal.forecast(len(y_test['LV ActivePower (kW)']))
holt_pred_s=holt_pred_s+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- Holt Winter Seasonal Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train['Date/Time'],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test['Date/Time'],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test['Date/Time'],holt_pred_s,'r--')
plt.xticks(rotation=90)
plt.show()
plt.title("Daily data- Holt Winter Seasonal Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],holt_pred_s[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
holt_error_s,mse_holt_s=[],[]
for i,j in zip(y_test['LV ActivePower (kW)'],holt_pred_s):
	holt_error_s.append(i-j)
	mse_holt_s.append((i-j)**2)
#ARIMA model
arima=sm.tsa.arima.ARIMA(y_train['LV ActivePower (kW)'],order=(1,1,7)).fit()
arima_pred=arima.forecast(len(y_test['LV ActivePower (kW)']))
arima_pred=arima_pred+np.mean(y_train['LV ActivePower (kW)'])
plt.title("All data- ARIMA Model")
plt.xlabel("Date")
plt.ylabel("LV Active Power")
plt.plot(y_train['Date/Time'],y_train['LV ActivePower (kW)'],'b--')
plt.plot(y_test['Date/Time'],y_test['LV ActivePower (kW)'],'g--')
plt.plot(y_test['Date/Time'],arima_pred,'r--')
plt.xticks(rotation=90)
plt.show()
plt.title("Daily data-ARIMA Model")
plt.plot(split_day[:105],split_lv[:105],'b-.')
plt.plot(split_day[:105],arima_pred[:105],'g-.')
plt.xticks(rotation=90)
plt.show()
arima_error,mse_arima=[],[]
for i,j in zip(y_test['LV ActivePower (kW)'],arima_pred):
	arima_error.append(i-j)
	mse_arima.append((i-j)**2)
#mean errors
print("Mean of error of Average Model",np.mean(av_er))
print("Mean of error of Naive Model",np.mean(nav_er))
print("Mean of error of Drift Model",np.mean(d_er))
print("Mean of error of SES Model",np.mean(s_er))
print("Mean of error of ARMA Model",np.mean(arma_error))
print("Mean of error of Holt Model",np.mean(holt_error))
print("Mean of error of Holt Seasonal Model",np.mean(holt_error_s))
print("Mean of error of ARIMA Model",np.mean(arima_error))
#variance errors
print("Variance of error of Average Model",np.var(av_er))
print("Variance of error of Naive Model",np.var(nav_er))
print("Variance of error of Drift Model",np.var(d_er))
print("Variance of error of SES Model",np.var(s_er))
print("Variance of error of ARMA Model",np.var(arma_error))
print("Variance of error of Holt Model",np.var(holt_error))
print("Variance of error of Holt Seasonal Model",np.var(holt_error_s))
print("Variance of error of ARIMA Model",np.var(arima_error))
# calc acf of errors:
acf_avg=code_collection.calc_acf(av_er,120)
acf_naive=code_collection.calc_acf(nav_er,120)
acf_drift=code_collection.calc_acf(d_er,120)
acf_ses=code_collection.calc_acf(s_er,120)
acf_arma=code_collection.calc_acf(arma_error,120)
acf_holt=code_collection.calc_acf(holt_error,120)
acf_holt_s=code_collection.calc_acf(holt_error_s,120)
acf_arima=code_collection.calc_acf(arima_error,120)
#plot acf of errors:
code_collection.plt_acf(acf_avg,"Average",120)
code_collection.plt_acf(acf_naive,"Naive",120)
code_collection.plt_acf(acf_drift,"Drift",120)
code_collection.plt_acf(acf_ses,"SES",120)
code_collection.plt_acf(acf_arma,"ARMA",120)
code_collection.plt_acf(acf_arima,"ARIMA",120)
code_collection.plt_acf(acf_holt,"Holt Winter",120)
code_collection.plt_acf(acf_holt_s,"Holt Winter Seasonal",120)
#printing MSE:
print("MSE of Average Base Model:",av_mse)
print("MSE of Drift Base Model:",d_mse)
print("MSE of Naive Base Model:",nav_mse)
print("MSE of SES:",s_mse)
print("MSE of ARMA process", np.mean(mse_arma))
print("MSE of Holt Winter",np.mean(mse_holt))
print("MSE of Holt Winter Seasonal",np.mean(mse_holt_s))
print("MSE of ARIMA process", np.mean(mse_arima))
#printing RMSE
print("RMSE of Average Base Model:",np.sqrt(av_mse))
print("RMSE of Drift Base Model:",np.sqrt(d_mse))
print("RMSE of Naive Base Model:",np.sqrt(nav_mse))
print("RMSE of SES:",np.sqrt(s_mse))
print("RMSE of ARMA process", np.sqrt(np.mean(mse_arma)))
print("RMSE of Holt Winter",np.sqrt(np.mean(mse_holt)))
print("RMSE of Holt Winter Seasonal",np.sqrt(np.mean(mse_holt_s)))
print("RMSE of ARIMA process", np.sqrt(np.mean(mse_arima)))
#q values
q_avg=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_avg)
print("Q value Average Model",q_avg)
q_naive=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_naive)
print("Q value Naive Model",q_naive)
q_drift=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_drift)
print("Q value Drift Model",q_drift)
q_ses=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_ses)
print("Q value SES Model",q_ses)
q_holt=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_holt)
print("Q value Holt Model",q_holt)
q_holt_s=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_holt_s)
print("Q value Holt Seasonal Model",q_holt_s)
q_arma=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_arma)
print("Q value ARMA Model",q_arma)
q_arima=code_collection.q_val(y_test['LV ActivePower (kW)'],acf_arima)
print("Q value ARIMA Model",q_arima)