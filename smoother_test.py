from pykalman import KalmanFilter
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pystan

A = np.array([[0, 1, 0],[0,0,1],[0,0,0]])
C = np.array([1,0,0])
Q = np.array([[0,0,0],[0,0,0],[0,0,1]])

dt = 1./500.

Ad = scipy.linalg.expm(A*dt)

sim_time = 5
T = int(np.round(sim_time / dt))

l1 = 0.7
l2 = 0.8
m1 = 1.5
m2 = 0.5
r1 = 5e-1
r2 = 2.5e-1
g = 9.81

params = {'l1':l1,
          'l2':l2,
          'm1':m1,
          'm2':m2,
          'r1':r1,
          'r2':r2,
          'g':g}

def robut_manipulator_2d(x, tau, params):
    theta1 = params['theta1']
    theta2 = params['theta2']
    theta3 = params['theta3']
    theta4 = params['theta4']
    theta5 = params['theta5']
    r1 = params['r1']
    r2 = params['r2']
    g = params['g']

    m11 = theta1 + 2 * theta2 * np.cos(x[1])
    m12 = theta3 + theta2 * np.cos(x[1])
    m22 = theta3
    mdet = m11 * m22 - m12 ** 2

    u1 = g * theta4 * np.cos(x[0] + x[1]) + theta5 * g * np.cos(x[0])
    u2 = theta4 * g * np.cos(x[0] + x[1])
    c2 = -2 * x[2]**2 * theta2 * np.sin(x[1]) - 2 * x[2] * x[3] * theta2 * np.sin(x[1])

    f1 = - u1 - r1 * x[2] + tau[0]
    f2 = c2 - u2 - r2 * x[3] + tau[1]

    dx = np.zeros((4))
    dx[0] = x[2]
    dx[1] = x[3]
    dx[2] = (m22 * f1 - m12 * f2) / mdet
    dx[3] = (m11 * f2 - m12 * f1) / mdet

    return dx


def fill_params(params):
    m1 = params['m1']
    m2 = params['m2']
    l1 = params['l1']
    l2 = params['l2']
    params['theta1'] = l2**2 * m2 + l1**2 * (m1 + m2)
    params['theta2'] = l1 * l2 * m2
    params['theta3'] = l2**2 * m2
    params['theta4'] = l2 * m2
    params['theta5'] = l1 * (m1 + m2)
    return params

params = fill_params(params)
x = np.zeros((4,T+1))

tau = np.zeros((2, T))
dx = np.zeros((4, T))

for t in range(T):
    tau[0, t] = np.exp(-0.4 * dt * t)
    tau[1, t] = np.exp(-0.5 * dt * t)
    dx[:, t] = robut_manipulator_2d(x[:,t], tau[:, t], params)
    x[:,t+1] = x[:,t] + dt * dx[:, t]


quantisation_interval = np.pi*2/8000/50
q = np.zeros(x[0:2, :].shape)
qs = np.zeros((2, T))
dqs = np.zeros((2, T))
ddqs = np.zeros((2, T))


q[:,0] = np.floor(x[0:2, 0] / quantisation_interval)*quantisation_interval
for t in range(1,T):
    dif = x[0:2, t+1] - q[:, t]
    s = np.sign(dif)
    add = np.floor(np.abs(dif)/quantisation_interval)*quantisation_interval
    q[:,t+1] = q[:,t] + s * add

kf = KalmanFilter(transition_matrices=Ad, observation_matrices=[[1,0,0]],observation_covariance=[[1e-5]])
measurements = q[0,:-1]
kf = kf.em(measurements, n_iter=30)

(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

qs[0,:] = smoothed_state_means[:,0]
dqs[0,:] = smoothed_state_means[:,1]
ddqs[0,:] = smoothed_state_means[:,2]

kf = KalmanFilter(transition_matrices=Ad, observation_matrices=[[1,0,0]],observation_covariance=[[1e-5]])
measurements = q[1,:-1]
kf = kf.em(measurements, n_iter=30)

(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

qs[1, :] = smoothed_state_means[:,0]
dqs[1, :] = smoothed_state_means[:,1]
ddqs[1, :] = smoothed_state_means[:,2]

model = pystan.StanModel(file='stan/robot_arm_2d.stan')

stan_data = {
    'N':T,
    'q':qs[:, :],
    'dq':dqs[:, :],
    'tau':tau[:, :],
    'ddq':ddqs[:, :],
    'g':g
}

control = {"adapt_delta": 0.85,
           "max_treedepth":13}

fit = model.sampling(data=stan_data, warmup=4000,iter=6000, control=control)

traces = fit.extract()

m1_traces = traces['m1']
m2_traces = traces['m2']
l1_traces = traces['l1']
l2_traces = traces['l2']
d1_traces = traces['d1']
d2_traces = traces['d2']
r_traces = traces['r']

plt.subplot(3,2,1)
plt.hist(m1_traces, density=True)
plt.axvline(m1, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$m_1$')

plt.subplot(3,2,2)
plt.hist(m2_traces, density=True)
plt.axvline(m2, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$m_2$')

plt.subplot(3,2,3)
plt.hist(l1_traces, density=True)
plt.axvline(l1, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$l_1$')

plt.subplot(3,2,4)
plt.hist(l2_traces, density=True)
plt.axvline(l2, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$l_2$')

plt.subplot(3,2,5)
plt.hist(d1_traces, density=True)
plt.axvline(r1, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$r_1$')

plt.subplot(3,2,6)
plt.hist(d2_traces, density=True)
plt.axvline(r2, linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$r_2$')

plt.tight_layout()
plt.show()


plt.hist(r_traces)
plt.show()



