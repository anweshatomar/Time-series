from FinalProjectFunctions import *
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# IMPORT DATA
df = pd.read_csv('T1.csv', header=0)
data = df.copy()
data['Date/Time']=pd.to_datetime(data['Date/Time'])

dep_var = data['LV ActivePower (kW)']
dep_var=dep_var[:5000]
train, test = train_test_split(dep_var, shuffle=False, test_size=0.2)
# print(train.shape, test.shape)  # (4146,) (1037,)
# print(train.head())
# print(test.head())
def step_0(na,nb):
    theta = np.zeros(shape=(na+nb,1))
    return theta.flatten()

def white_noise_simulation(theta,na,y):
    num = [1] + list(theta[na:])
    den = [1] + list(theta[:na])
    while len(num) < len(den):
        num.append(0)
    while len(num) > len(den):
        den.append(0)
    system = (den, num, 1)
    tout, e = signal.dlsim(system, y)
    e = [a[0] for a in e]
    return np.array(e)

def step_1(theta,na,nb,delta,y):
    e = white_noise_simulation(theta,na,y)
    SSE = np.matmul(e.T, e)
    X_all = []
    for i in range(na+nb):
        theta_dummy = theta.copy()
        theta_dummy[i] = theta[i] + delta
        e_n = white_noise_simulation(theta_dummy,na,y)
        X_i = (e - e_n)/delta
        X_all.append(X_i)

    X = np.column_stack(X_all)
    A = np.matmul(X.T,X)
    g = np.matmul(X.T,e)
    return A,g,SSE

def step_2(A,mu,g,theta,na,y):
    I = np.identity(g.shape[0])
    theta_d = np.matmul(np.linalg.inv(A+(mu*I)),g)
    theta_new = theta + theta_d
    e_new = white_noise_simulation(theta_new,na,y)
    SSE_new = np.matmul(e_new.T,e_new)
    if np.isnan(SSE_new):
        SSE_new = 10 ** 10
    return SSE_new, theta_d, theta_new


with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

def step_3(max_iterations, mu_max, na, nb, y, mu, delta):
    iteration_num = 0
    SSE = []
    theta = step_0(na, nb)
    while iteration_num < max_iterations:
        print('Iteration ', iteration_num)

        A, g, SSE_old = step_1(theta, na, nb, delta, y)
        print('old SSE : ', SSE_old)
        if iteration_num == 0:
            SSE.append(SSE_old)
        SSE_new, theta_d, theta_new = step_2(A, mu, g, theta, na, y)
        print('new SSE : ', SSE_new)
        SSE.append(SSE_new)

        if SSE_new < SSE_old:
            print('Norm of delta_theta :', np.linalg.norm(theta_d))
            if np.linalg.norm(theta_d) < 1e-3:
                theta_hat = theta_new
                e_var = SSE_new / (len(y) - A.shape[0])
                cov = e_var * np.linalg.inv(A)
                print('\n **** Algorithm Converged **** \n')
                return SSE, theta_hat, cov, e_var
            else:
                theta = theta_new
                mu = mu / 10

        while SSE_new >= SSE_old:
            mu = mu * 10
            if mu > mu_max:
                print('mu exceeded the max limit')
                return None, None, None, None
            SSE_new, theta_d, theta_new = step_2(A, mu, g, theta, na, y)

        theta = theta_new

        iteration_num+=1
        if iteration_num > max_iterations:
            print('Max iterations reached')
            return None, None, None, None


np.random.seed(10)
mu_factor = 10
delta = 1e-6
epsilon = 0.001
mu = 0.01
max_iterations = 100
mu_max = 1e10

na = 1
nb = 7

SSE, est_params, cov, e_var = step_3(max_iterations, mu_max, na, nb, train, mu, delta)
print('Estimated parameters : ', est_params)
print('Estimated Covariance matrix : ', cov)
print('Estimated variance of error : ', e_var)

def SSEplot(SSE):
    plt.figure()
    plt.plot(SSE, label = 'Sum Squared Error')
    plt.xlabel('# of Iterations')
    plt.ylabel('Sum Squared Error')
    plt.legend()
    plt.show()

SSEplot(SSE)

