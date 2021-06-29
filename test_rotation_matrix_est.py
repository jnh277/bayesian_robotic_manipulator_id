import numpy as np
import pystan
import matplotlib.pyplot as plt

num_tests = 1
sigma = 1e-4        # measurement standard deviation

euler_angles = np.random.uniform(0, 2 * np.pi, (num_tests, 3))

def eulerRotation(phi, theta, psi):

    rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                  [0, 0, 1]])

    ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])

    r = rz @ ry @ rx

    return r

r = np.zeros((num_tests, 3, 3))

for i in range(num_tests):
    r[i, :, :] = eulerRotation(euler_angles[i, 0], euler_angles[i, 1], euler_angles[i, 2])

axis_vectors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

y = r @ axis_vectors + np.random.normal(0., sigma, (num_tests,3,3))

# perform estimation

model = pystan.StanModel(file='stan/rotation.stan')

num_test = 0
stan_data = {
    'N':3,
    'axis_vectors':axis_vectors,
    'y':y[num_test,:,:]
}

fit = model.sampling(data=stan_data)

traces = fit.extract()

sigma_hat = traces['r']
r_hat = traces['R']
y_hat = traces['yhat']

plt.hist(sigma_hat)
plt.show()