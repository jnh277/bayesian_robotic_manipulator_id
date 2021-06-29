import pystan
import numpy as np
import matplotlib.pyplot as plt


sim_time = 200
dt = 1/10
T = int(np.round(sim_time / dt))


def sim(x, u, a=1, b=1):
    dx = - a * x + b * u
    return dx

x = np.zeros((1, T+1))
x[0, 0] = 1.0
u = np.zeros((1, T))
dx = np.zeros((1, T))
for t in range(T):
    if not t % 5:
        u[0, t] = np.random.uniform(-1, 1)
    else:
        u[0, t] = u[0,t-1]
    dx[:, t] = sim(x[0, t], u[:,t])
    x[0, t+1] = x[0, t] + dt * dx[:, t]


plt.plot(np.arange(T+1) * dt, x[0, :])
plt.show()


model = pystan.StanModel(file='stan/simple_example.stan')

stan_data = {
    'N':T,
    'x':x[0,:-1],
    'dx':dx[0,:],
    'u':u[0,:]
}

fit = model.sampling(data=stan_data)

traces = fit.extract()


r = traces['r']
a = traces['a']
b = traces['b']

plt.subplot(2,2,1)
plt.hist(a)

plt.subplot(2,2,2)
plt.hist(b)

plt.subplot(2,2,3)
plt.hist(r)

plt.tight_layout()
plt.show()
