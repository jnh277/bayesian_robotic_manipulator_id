
import pystan
import numpy as np
# import matplotlib
# matplotlib.rc('axes.formatter', useoffset=False)
import matplotlib.pyplot as plt
import scipy.signal as signal

import seaborn as sns

sim_time = 5
dt = 1/500
T = int(np.round(sim_time / dt))
quantisation_interval = np.pi*2/8000/50

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

def robot_manipulator_2d(x, tau, params):
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

    # print(u1)
    # print(u2)
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

def rk4_dx(x, tau, params, dt):
    k1 = robot_manipulator_2d(x, tau, params)
    k2 = robot_manipulator_2d(x+dt*k1/2, tau, params)
    k3 = robot_manipulator_2d(x+dt*k2/2, tau, params)
    k4 = robot_manipulator_2d(x+dt*k3, tau, params)
    return (k1/6 + k2/3 + k3/3 + k4/6)


params = fill_params(params)
x = np.zeros((4,T+1))
# x[0, 0] = np.pi/2
# x[1, 0] = np.pi/2

tau = np.zeros((2, T))
dx = np.zeros((4, T))
dx2 = 1.0*dx

# T = 1
for t in range(T):
    if not t % 50:
        tau[:, t] = np.random.uniform(-1,1,(2,))
    else:
        tau[:, t] = tau[:, t-1]
    # tau[0, t] = np.exp(-0.4 * dt * t)
    # tau[1, t] = np.exp(-0.5 * dt * t)
    dx[:, t] = rk4_dx(x[:,t], tau[:, t], params, dt)
    dx2[:, t] = robot_manipulator_2d(x[:,t], tau[:, t], params)
    x[:,t+1] = x[:,t] + dt * dx[:, t]

# 1.0/0.0
plt.subplot(2,1,1)
plt.plot(dt * np.arange(T),tau[0,:])

plt.subplot(2,1,2)
plt.plot(dt * np.arange(T),tau[1,:])
plt.tight_layout()
plt.show()

plt.subplot(4,1,1)
plt.plot(dt * np.arange(T+1), x[0,:])

plt.subplot(4,1,2)
plt.plot(dt * np.arange(T+1), x[1,:])

plt.subplot(4,1,3)
plt.plot(dt * np.arange(T+1), x[2,:])

plt.subplot(4,1,4)
plt.plot(dt * np.arange(T+1), x[3,:])
plt.tight_layout()
plt.show()


model = pystan.StanModel(file='stan/robot_arm_2d.stan')

# q = x[0:2, :-1]
# dq = np.vstack((np.gradient(q[0,:]),  np.gradient(q[1,:])))/ dt
# ddq = np.vstack((np.gradient(dq[0,:]),  np.gradient(dq[1,:])))/ dt
#
# for v1


# q = x[0:2, :]+np.random.normal(0.0,1e-6,(2,T+1)) # rough ball park for quantisation variance
q = np.zeros(x[0:2, :].shape)
q[:,0] = np.floor(x[0:2, 0] / quantisation_interval)*quantisation_interval
for t in range(1,T):
    dif = x[0:2, t+1] - q[:, t]
    s = np.sign(dif)
    add = np.floor(np.abs(dif)/quantisation_interval)*quantisation_interval
    q[:,t+1] = q[:,t] + s * add

# forward difference
# dq = (q[:,1:]-q[:,:-1])/dt
# ddq = (dq[:,1:]-dq[:,:-1])/dt
#
# stan_data = {
#     'N':T-1,
#     'q':q[:, :-2],
#     'dq':dq[:, :-1],
#     'tau':tau[:, :-1],
#     'ddq':ddq[:, :],
#     'g':g
# }

# alternate savitzky-golay filter
dq = signal.savgol_filter(q, 51, 5, deriv=1, delta=dt, axis=1)
ddq = signal.savgol_filter(q, 51, 5, deriv=2, delta=dt, axis=1)

stan_data = {
    'N':T,
    'q':q[:, :-1],
    'dq':dq[:, :-1],
    'tau':tau[:, :],
    'ddq':ddq[:, :-1],
    'g':g
}

# for v2
# q = x[0:2, :]
# dq = (q[:,1:]-q[:,:-1])/dt
#
# stan_data = {
#     'N':T,
#     'q':q[:, :-1],
#     'dq':dq[:, :],
#     'tau':tau[:, :],
#     'g':g,
#     'dt':dt,
# }


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

nu = traces['nu']
plt.hist(nu, bins=30)
plt.show()


ddq1hat = traces['ddq1hat']

plt.plot(ddq1hat.mean() - ddq[0,:-1])
plt.show()

# bias = traces['bias']
#
# plt.subplot(2,1,1)
# plt.hist(bias[:, 0], bins=30)
#
# plt.subplot(2,1,2)
# plt.hist(bias[:, 1], bins=30)
# plt.show()