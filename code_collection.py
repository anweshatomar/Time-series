import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg
from scipy import signal
#mean of values for lab1
def calc_mean(data):
    sum_sales = 0
    m_sales = []
    v_sales = []
    temp = pd.Series()
    n = 1
    for s in data['Sales']:
        sum_sales = sum_sales + s
        mean = sum_sales / n
        temp = data['Sales'][:n]
        variance = temp.var()
        m_sales.append(mean)
        v_sales.append(variance)
        n += 1
    sum_gdp = 0
    m_gdp = []
    v_gdp = []
    temp = pd.Series()
    n = 1
    for g in data['GDP']:
        sum_gdp = sum_gdp + g
        mean = sum_gdp / n
        temp = data['GDP'][:n]
        variance = temp.var()
        m_gdp.append(mean)
        v_gdp.append(variance)
        n += 1
    sum_adb = 0
    m_adb = []
    v_adb = []
    temp = pd.Series()
    n = 1
    for a in data['AdBudget']:
        sum_adb = sum_adb + a
        mean = sum_adb / n
        temp = data['AdBudget'][:n]
        variance = temp.var()
        m_adb.append(mean)
        v_adb.append(variance)
        n += 1
    return m_sales,v_sales, m_gdp, v_gdp, m_adb, v_adb
#correlation coefficient
def correlation_coefficent_cal(x,y):
    mean1=np.mean(x)
    mean2=np.mean(y)
    sum1=np.sum((x-mean1)*(y-mean2))
    sum2=np.sqrt(np.sum((x-mean1)**2))
    sum3=np.sqrt(np.sum((y-mean2)**2))
    val2=sum2*sum3
    r =sum1/val2
    return r
# ACF calculations
def calc_tow(Y,tow):
    l=len(Y)
    num=0
    den=0
    y_mean=np.mean(Y)
    for s in Y:
        den=den+((s-y_mean)**2)
    for j in range(tow,l):
        v1=Y[j] - y_mean
        v2=Y[j-tow] - y_mean
        num=num+(v1*v2)
    val=num/den
    return val
def calc_acf(data,lags):
    val = []
    for i in range(0, lags):
        val.append(calc_tow(data, i))
    val_w = val[1:]
    r_val = val_w[::-1]
    acf_val= r_val + val
    return acf_val
def plt_acf(acf_set,lb,lags):
    plt.stem(np.arange(-lags,lags-1),acf_set, label=lb)
    plt.title("Autocorrelation plot")
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.show()


# ---------------------------------
#  Generate y - auxiliary functions
# ---------------------------------
# Input coefficients based on order
def process_coef(order, process):
    coefficients = []
    for i in range(1, order + 1):
        coef = float(input("Enter coefficient #{} of {} process (ex: -.5 or .8) = ".format(i, process)))
        coefficients.append(coef)
    return coefficients


# den and num should have the same size
# function returns [1, a1, a2,...]
def prepare_dlsim_coef(a, b):
    # a = den = ar_coef = y
    # b = num = ma_coef = e
    # Evaluate size
    if type(a) != list or type(b) != list:
        a = a.tolist()
        b = b.tolist()
    if len(a) > len(b):
        while len(a) > len(b):
            b.append(0.0)
    elif len(b) > len(a):
        while len(b) > len(a):
            a.append(0.0)
    # add 1
    den_dlsim = [1]
    den_dlsim.extend(a)
    num_dlsim = [1]
    num_dlsim.extend(b)
    return den_dlsim, num_dlsim


# generate y with dlsim
def dlsim_generate_y(e, den, num):
    den = den  # na
    num = num  # nb
    sys = (num, den, 1)
    tout, y = signal.dlsim(sys, e)
    return y.flatten()


# convert y (in case mean!=0 and var!=1)
def convert_y(y, mean, var, ar_coef, ma_coef):
    if mean != 0 and var != 1:
        mean_y = (mean * (1 + np.sum(ma_coef))) / (1 + np.sum(ar_coef))
        y_final = y - mean_y
        print("Mean is different from 0 and Var is different from 1. Data was converted.")
    else:
        y_final = y
    return y_final


# ---------------------------------
#   Generate y - MAIN fuction
# ---------------------------------
# input variables
def generate_y():
    # ---------- input variables -------------
    T = int(input("Enter number of data samples = "))
    mean = int(input("Enter the mean of white noise = "))
    var = int(input("Enter the variance of white noise = "))
    std = np.sqrt(var)
    na = int(input("Enter the AR order = "))
    process = "AR"
    ar_coef = process_coef(na, process)
    nb = 1
    nb = int(input("Enter the MA order = "))
    process = "MA"
    ma_coef = process_coef(nb, process)
    print("AR coefficients = ", ar_coef)
    print("Ma coefficients = ", ma_coef)

    # ---------- Generate y -------------
    # prepare dlsim coefficients
    ar_coef_dlsim, ma_coef_dlsim = prepare_dlsim_coef(a=ar_coef, b=ma_coef)
    # Define e
    e_orig = np.random.normal(mean, std, size=T)
    # Generate y
    y_orig = dlsim_generate_y(e=e_orig, den=ar_coef_dlsim, num=ma_coef_dlsim)
    # Convert y
    y = convert_y(y=y_orig, mean=mean, var=var, ar_coef=ar_coef, ma_coef=ma_coef)

    return na, nb, ar_coef, ma_coef, y


# def data_generation():
# input variables and generate y
#    na,nb,ar_coef,ma_coef,y = generate_y()
#    return na,nb,ar_coef,ma_coef,y

# ---------------------------------
#     LM - auxiliary functions
# ---------------------------------
# hyperparameters
iterations = 10
delta = pow(10, -6)
miu = 0.01
miu_max = pow(10, 10)
max_iterations = 50


# Calculate e
def cal_e(na, nb, y, theta):
    den_orig = theta[:na]  # ar
    num_orig = theta[na:]  # ma
    # prepare dlsim coefficients
    den_dlsim, num_dlsim = prepare_dlsim_coef(a=den_orig, b=num_orig)
    # generate e with dlsim
    den = den_dlsim  # na
    num = num_dlsim  # nb
    sys = (den, num, 1)
    tout, e = signal.dlsim(sys, y)
    return e.flatten()


# Calculate SSE
def cal_SSE(e):
    # e transposed
    e_transposed = e.T
    # calculate SSE
    SSE = e.dot(e_transposed)
    return SSE


# calculate negative gradient
def cal_gradient(na, nb, y, e, delta, theta):
    num_parameters = na + nb
    x_matrix_values = []
    x = np.zeros(len(e))
    for i in range(0, num_parameters):
        theta[i] += delta
        e_new = cal_e(na, nb, y, theta)
        x = (e - e_new) / delta
        x_matrix_values.append(x)
        # subtract delta
        theta[i] -= delta
    X = np.stack(x_matrix_values, axis=1)
    # Calculate A
    A = X.T.dot(X)
    # Calculate g
    g = X.T.dot(e)
    return A, g


# Calculate theta change
def cal_delta_theta(na, nb, miu, A, g):
    num_parameters = na + nb
    theta_change = g.dot(np.linalg.inv((np.identity(num_parameters) * miu) + A))
    return theta_change


# ------------ step 1 ------------
def step1(na, nb, y, delta, theta):
    # calculate e
    e = cal_e(na, nb, y, theta)
    # calculate SSE
    SSE_old = cal_SSE(e=e)
    # calculate negative gradient
    A, g = cal_gradient(na, nb, y, e, delta, theta)
    return SSE_old, A, g


# ------------ step 2 ------------
def step2(na, nb, y, miu, A, g, theta):
    # calcultae delya change
    delta_theta = cal_delta_theta(na, nb, miu, A, g)
    # add cange to theta
    theta_new = theta + delta_theta
    # calculate new e based on change
    e_new = cal_e(na, nb, y, theta=theta_new)
    # calcultae new SSE based on change
    SSE_new = cal_SSE(e=e_new)
    return SSE_new, theta_new, delta_theta


# calculate confidence intervals
def cal_conf_interval(theta_hat, cov_theta_hat):
    coef = theta_hat
    a = coef - (2 * (np.sqrt(cov_theta_hat)))
    b = coef + (2 * (np.sqrt(cov_theta_hat)))
    conf = np.concatenate((a, b), axis=None)
    return conf


# ---------------------------------
#       LM - MAIN function
# ---------------------------------
def LM_algorithm():
    na, nb, ar_coef, ma_coef, y = generate_y()
    y = np.array(y)

    # hyperparameters
    # iterations = 50
    delta = pow(10, -6)
    miu = 0.01
    miu_max = pow(10, 10)
    max_iterations = 50

    SSE_list = []

    # initializse thetas to zero
    theta_init = [0.0 for a in range(1, na + 1)] + [0.0 for b in range(1, nb + 1)]
    SSE_old, A, g = step1(na, nb, y, delta, theta=theta_init)
    SSE_new, theta_new, delta_theta = step2(na, nb, y, miu, A, g, theta=theta_init)
    SSE_list.append(SSE_new)

    iterations = 1
    # if iterations < max_iterations:
    for i in range(1, max_iterations):

        if np.isnan(SSE_new):
            SSE_new = np.exp(10)
        else:
            if SSE_new < SSE_old:
                if linalg.norm(np.array(delta_theta), ord=2) < pow(10, -3):
                    # print("***************Algorithm converges ******************")
                    # algorithm converges because there is no significant contribution of delta theta
                    theta_hat = theta_new
                    # variance of error
                    var_error = SSE_new / (len(y) - (na + nb))
                    # cov_theta
                    cov_theta_hat = var_error * linalg.inv(A)  # for conf intervals
                    break
                else:
                    theta_old = theta_new
                    miu = miu / 10  # decrease miu

            while SSE_new > SSE_old:
                miu = miu * 10  # increase miu
                # print("increase miu", miu)
                if miu > miu_max:
                    print("ERROR: miu exceeds maximum.")
                    break
                # change miu
                theta_old = theta_new
                SSE_new, theta_new, delta_theta = step2(na, nb, y, miu, A, g, theta=theta_old)

            # theta = theta_new
            SSE_old, A, g = step1(na, nb, y, delta, theta=theta_old)
            SSE_new, theta_new, delta_theta = step2(na, nb, y, miu, A, g, theta=theta_old)
            SSE_list.append(SSE_new)
            iterations += 1

    plt.figure()
    plt.plot(range(0, iterations), np.array(SSE_list))
    plt.title("SSE vs number of iterations")
    plt.show()

    # elif iterations > max_iterations:
    #    print("ERROR: iterations exceed maximum.")
    print("True parameters are = ", ar_coef, ma_coef)
    print("Estimated parameters are = ", theta_hat)
    print("Variance is = ", var_error)
    print("Covariance Matrix is = ", cov_theta_hat)
    # print("iterations = " ,iterations)

    # calculate confidence intervals
    conf_intervals = []
    for i in range(len(theta_hat)):
        intervals = cal_conf_interval(theta_hat[i], cov_theta_hat[i][i])
        conf_intervals.append(intervals)
    # print confidence intervals of a coefficients
    for i in range(len(conf_intervals[:na])):
        print(conf_intervals[:na][i][0], " < a{} < ".format(i + 1), conf_intervals[:na][i][1])
    # print confidence intervals of b coefficients
    for i in range(len(conf_intervals[na:])):
        print(conf_intervals[na:][i][0], " < b{} < ".format(i + 1), conf_intervals[na:][i][1])

    return theta_hat, var_error, cov_theta_hat, conf_intervals, y, na, nb, ar_coef, ma_coef


# /////////////////////////////////////////// #
# ----- Autocorrelation Function ----------- #
# /////////////////////////////////////////// #
def ACF_cal(y, lags, title):
    # convert to array
    array = np.array(y)
    # get mean
    mean = sum(y) / len(y)
    # length
    samples = len(y)
    # subtract mean to each value
    x = array - mean
    # Calculate Denominator
    denominator = x.dot(x)
    # Calculate Taus
    tau = np.zeros(lags + 1)
    tau[0] = 1  # The first tau is 1
    # Create a loop that iterates all lags (except lag 0)
    for i in range(lags):
        tau[i + 1] = x[i + 1:].dot(x[:-(i + 1)])
    # divide by denominator
    tau[1:lags + 1] = tau[1:lags + 1] / denominator

    # plot
    coordinates = []
    for i in range(len(tau)):
        coordinates.append((i, tau[i]))
        coordinates.append((i * -1, tau[i]))

    axis_x = []
    axis_y = []
    for pair in coordinates:
        axis_x.append(pair[0])
        axis_y.append(pair[1])

    # plot
    plt.figure(figsize=(8, 5))
    plt.title("Autocorrelation of {} (lags={} and samples={})".format(title, lags, samples),
              fontsize=14)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.stem(axis_x, axis_y)
    plt.show()

    r = tau.copy()

    return r
def cal_gpac(r, j_scope, k_scope):
    gpac = np.zeros(shape=(j_scope, k_scope), dtype=np.float64)
    for j in range(j_scope):
        for k in range(1, k_scope+1):
            den = np.zeros(shape=(k, k), dtype='int64')
            for i in range(k):
                den[:, i] = np.arange(j - i, j + k - i)
            den = np.abs(den)
            den_acf = np.take(r, den)
            den_deter = np.linalg.det(den_acf)

            num = den
            num[:, -1] = den[:, 0] + 1
            num = np.abs(num)
            num_acf = np.take(r, num)
            num_deter = np.linalg.det(num_acf)

            phi_j_k = np.round(np.divide(num_deter, den_deter), 5)

            gpac[j, k-1] = phi_j_k
    return gpac
def GPAC_matrix(R,j,k):
    num=np.zeros((k,k))
    den=np.zeros((k,k))
    for a1 in range(0,k):
        for a2 in range(0,k):
            #diagonal elements
            if a1==a2:
                num[a1][a2]=R[j]
                den[a1][a2] = R[j]
            #last diagonal element
            if a1==k-1 and a2==k-1:
                num[a1][a2] = R[j+k]
                den[a1][a2] = R[j]
            #a1=0 a2=1
            if a1>a2:
                num[a1][a2] = R[j + a1]
                den[a1][a2] = R[j + a1]
            # a2=0 a1=1
            if a2 > a1:
                num[a1][a2] = R[j - a2]
                den[a1][a2] = R[j - a2]
            #last row
            if a1==k-1:
                num[a1][a2] = R[j +k - a2-1]
                den[a1][a2] = R[j + k - a2 - 1]
            #last column
            if a2==k-1:
                num[a1][a2] = R[j+ a1 +1]
                den[a1][a2] = R[j -k+a1+1]
    final=np.linalg.det(num)/np.linalg.det(den)
    return final
def GPAC(r,j,k):
    arr = np.zeros((j, k))
    for a1 in range(0,j):
        for a2 in range(0,k):
            arr[a1][a2]=GPAC_matrix(r, a1, a2)
    return arr
def adj_seasonality(data,y):
    y_hat=[]
    for i, j in zip(data,y):
        y_hat.append(i-j)
    return y_hat


# for adding zeros to get the strength of seasonality addition
def modify(mva,fold):
    l=[0]*fold
    b=[0]*fold
    final=l+mva+b
    return final



def test_input(m,k):
    if (m>=3):
        if (m%2!=0):
            if (k>0 and k%2==0):
                print("Wrong input, both should be odd values.")
                return False
        elif(m%2==0):
            if (k%2!=0):
                print("Wrong Input, both should be even values.")
                return False
    else:
        print("Input moving average value greater than 2.")
        return False
def adf_test(data):
    X = data.values
    result = adfuller(X)
    print(result[0])
#function for folding
def folding(data,fold):
    k=int((fold-1)/2)
    mva = []
    test = len(data) - 2 * (k+1)
    for i in range(0, len(data)):
        if (k <= test):
            temp = i + fold
            val = sum(data[i:temp]) / fold
            mva.append(val)
            k = k + 1
    return mva


#intitial function
def mv_avg(m,fold,data):
    k=int((m-1)/2)
    mva = []
    test=len(data)-2*k
    for i in range(0,len(data)):
        if (k <=test):
            temp=i+m
            val=sum(data[i:temp])/m
            mva.append(val)
            k=k+1
    if (m%2==0):
        final=folding(mva,fold)
        return final
    else:
        return mva
def q_val(train_set, acf):
    T = len(train_set)
    sum= 0
    for val in acf[1:]:
        sum += (val**2)
    Q = T*sum
    return sum
#calculation avg
def calc_avg(train,test):
    final_avg=[]
    error=[]
    error2=[]
    avg=np.mean(train)
    for i in range(0,len(test)):
        final_avg.append(avg)
    for t,f in zip(train,final_avg):
        error.append(t-f)
    for e in error:
        error2.append(e**2)
    mse=np.mean(error2)
    return final_avg,error,mse
#calculating naive
def calc_naive(train,test):
    train=list(train)
    test=list(test)
    j=0
    naive = []
    error = []
    error2 = []
    for t in range(0,len(test)):
        naive.append(train[-1])
        j+=1
    for t1,t2 in zip(test,naive):
        error.append(t1-t2)
    for e in error:
        error2.append(e**2)
    mse=np.mean(error2)
    return naive,error,mse
#calculations for drift
def calc_drift(train,test):
    train=list(train)
    test=list(test)
    yt=train[-1]
    y1=train[0]
    T=len(train)
    error=[]
    error2=[]
    final_val=[]
    for j in range(1,len(test)):
        num=j*((yt-y1)/(T-1))
        final=yt+num
        final_val.append(final)
    for t,f in zip(train,final_val):
        error.append(t-f)
        error2.append((t-f)**2)
    mse = np.mean(error2)
    return final_val,error,mse
#caculations for ses
def calc_ses(alfa,train,test):
    ses=[]
    error=[]
    error2=[]
    ses.append(train[-1])
    for j in range(1,len(test)):
        ses.append(train[-1])
    for i, j in zip(test,ses):
        error.append(i-j)
        error2.append((i-j)**2)
    mse = np.mean(error2)
    return ses,error,mse