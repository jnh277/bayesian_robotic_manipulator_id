import numpy as np
import pystan
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import time
import os
from tqdm import tqdm

sigmas = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
num_tests = 30


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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


# load / compile models
model_name = 'rotation'
if Path('stan/' + model_name + '.pkl').is_file():
    quarternion_model = pickle.load(open('stan/' + model_name + '.pkl', 'rb'))
else:
    quarternion_model = pystan.StanModel(file='stan/' + model_name + '.stan', )
    with open('stan/' + model_name + '.pkl', 'wb') as file:
        pickle.dump(quarternion_model, file)

model_name = 'rotation_euler'
if Path('stan/' + model_name + '.pkl').is_file():
    euler_model = pickle.load(open('stan/' + model_name + '.pkl', 'rb'))
else:
    euler_model = pystan.StanModel(file='stan/' + model_name + '.stan', )
    with open('stan/' + model_name + '.pkl', 'wb') as file:
        pickle.dump(euler_model, file)

model_name = 'rotation_CK'
if Path('stan/' + model_name + '.pkl').is_file():
    CK_model = pickle.load(open('stan/' + model_name + '.pkl', 'rb'))
else:
    CK_model = pystan.StanModel(file='stan/' + model_name + '.stan', )
    with open('stan/' + model_name + '.pkl', 'wb') as file:
        pickle.dump(CK_model, file)

model_name = 'rotation_6d'
if Path('stan/' + model_name + '.pkl').is_file():
    d6_model = pickle.load(open('stan/' + model_name + '.pkl', 'rb'))
else:
    d6_model = pystan.StanModel(file='stan/' + model_name + '.stan', )
    with open('stan/' + model_name + '.pkl', 'wb') as file:
        pickle.dump(d6_model, file)

num_sigmas = len(sigmas)

axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
axis_vectors = np.array([[1, 0], [0, 1], [0, 0]])
r = np.zeros((num_sigmas, num_tests, 3, 3))
y = np.zeros((num_sigmas, num_tests, 3, 2))

cond_mean_err_q = np.zeros((num_sigmas, num_tests, 3, 3))
cond_mean_err_e = np.zeros((num_sigmas, num_tests, 3, 3))
cond_mean_err_CK = np.zeros((num_sigmas, num_tests, 3, 3))
cond_mean_err_6d = np.zeros((num_sigmas, num_tests, 3, 3))

sample_var_err_q = np.zeros((num_sigmas, num_tests, 3, 3))
sample_var_err_e = np.zeros((num_sigmas, num_tests, 3, 3))
sample_var_err_CK = np.zeros((num_sigmas, num_tests, 3, 3))
sample_var_err_6d = np.zeros((num_sigmas, num_tests, 3, 3))

fit_times_q = np.zeros((num_sigmas, num_tests,))
fit_times_e = np.zeros((num_sigmas, num_tests,))
fit_times_CK = np.zeros((num_sigmas, num_tests,))
fit_times_6d = np.zeros((num_sigmas, num_tests,))

# sigma = 1e-4        # measurement standard deviation
euler_angles = np.random.uniform(0, 2 * np.pi, (num_sigmas, num_tests, 3))

for j, sigma in enumerate(sigmas):
    for i in tqdm(range(num_tests), desc='Running trials for noise std ' + str(sigma)):
        r[j, i, :, :] = eulerRotation(euler_angles[j, i, 0], euler_angles[j, i, 1], euler_angles[j, i, 2])

        y[j, i] = r[j, i, :, :] @ axis_vectors + np.random.normal(0., sigma, (3, 2))

        stan_data = {
            'N': 2,
            'axis_vectors': axis_vectors,
            'y': y[j, i, :, :]
        }

        # perform estimation
        t0 = time.time()
        with suppress_stdout_stderr():
            fit = quarternion_model.sampling(data=stan_data)
        t1 = time.time()
        fit_times_q[j, i] = t1 - t0
        quarternion_traces = fit.extract()

        t0 = time.time()
        with suppress_stdout_stderr():
            fit = euler_model.sampling(data=stan_data)
        t1 = time.time()
        fit_times_e[j, i] = t1 - t0
        euler_traces = fit.extract()

        t0 = time.time()
        with suppress_stdout_stderr():
            fit = d6_model.sampling(data=stan_data)
        t1 = time.time()
        fit_times_6d[j, i] = t1 - t0
        d6_traces = fit.extract()

        t0 = time.time()
        with suppress_stdout_stderr():
            fit = CK_model.sampling(data=stan_data)
        t1 = time.time()
        fit_times_CK[j, i] = t1 - t0
        CK_traces = fit.extract()

        rhat_q = quarternion_traces['R']
        rhat_e = euler_traces['R']
        rhat_6d = d6_traces['R']
        rhat_CK = CK_traces['R']

        cond_mean_err_q[j, i, :, :] = r[j, i, :, :] - rhat_q.mean(axis=0)
        cond_mean_err_e[j, i, :, :] = r[j, i, :, :] - rhat_e.mean(axis=0)
        cond_mean_err_CK[j, i, :, :] = r[j, i, :, :] - rhat_CK.mean(axis=0)
        cond_mean_err_6d[j, i, :, :] = r[j, i, :, :] - rhat_6d.mean(axis=0)

        sample_var_err_q[j, i, :, :] = np.mean((np.expand_dims(r[j, i, :, :], 0) - rhat_q) ** 2, axis=0)
        sample_var_err_e[j, i, :, :] = np.mean((np.expand_dims(r[j, i, :, :], 0) - rhat_e) ** 2, axis=0)
        sample_var_err_6d[j, i, :, :] = np.mean((np.expand_dims(r[j, i, :, :], 0) - rhat_6d) ** 2, axis=0)
        sample_var_err_CK[j, i, :, :] = np.mean((np.expand_dims(r[j, i, :, :], 0) - rhat_CK) ** 2, axis=0)

    plt.subplot(2, 2, 1)
    plt.hist(cond_mean_err_q[j].flatten(), bins=30, density=True)
    plt.xlabel('quarternion conditional mean err, sigma = ' + str(sigma))

    plt.subplot(2, 2, 2)
    plt.hist(cond_mean_err_e[j].flatten(), bins=30, density=True)
    plt.xlabel('euler conditional mean err, sigma = ' + str(sigma))

    plt.subplot(2, 2, 3)
    plt.hist(cond_mean_err_CK[j].flatten(), bins=30, density=True)
    plt.xlabel('CK conditional mean err, sigma = ' + str(sigma))

    plt.subplot(2, 2, 4)
    plt.hist(cond_mean_err_6d[j].flatten(), bins=30, density=True)
    plt.xlabel('6d conditional mean err, sigma = ' + str(sigma))

    plt.tight_layout()
    plt.show()

    plt.subplot(2, 2, 1)
    plt.hist(sample_var_err_q[j].flatten(), bins=30, density=True)
    plt.xlabel('quarternion conditional sample var, sigma = ' + str(sigma))

    plt.subplot(2, 2, 2)
    plt.hist(sample_var_err_e[j].flatten(), bins=30, density=True)
    plt.xlabel('euler conditional sample var, sigma = ' + str(sigma))

    plt.subplot(2, 2, 3)
    plt.hist(sample_var_err_CK[j].flatten(), bins=30, density=True)
    plt.xlabel('CK conditional sample var, sigma = ' + str(sigma))

    plt.subplot(2, 2, 4)
    plt.hist(sample_var_err_6d[j].flatten(), bins=30, density=True)
    plt.xlabel('6d conditional sample var, sigma = ' + str(sigma))

    plt.tight_layout()
    plt.show()

print('Mean absolute error of conditional mean')
print('quarternion:  ', np.mean(np.abs(cond_mean_err_q)))
print('euler:        ', np.mean(np.abs(cond_mean_err_e)))
print('CK:           ', np.mean(np.abs(cond_mean_err_CK)))
print('6d:           ', np.mean(np.abs(cond_mean_err_6d)))
print('\n')
print('Mean sample error variance')
print('quarternion:  ', np.mean(np.abs(sample_var_err_q)))
print('euler:        ', np.mean(np.abs(sample_var_err_e)))
print('CK:           ', np.mean(np.abs(sample_var_err_CK)))
print('6d:           ', np.mean(np.abs(sample_var_err_6d)))
print('\n')
print('Mean fitting times')
print('quarternion:  ', np.mean(fit_times_q))
print('euler:        ', np.mean(fit_times_e))
print('CK:           ', np.mean(fit_times_CK))
print('6d:           ', np.mean(fit_times_6d))

plt.subplot(3, 1, 1)
plt.loglog(sigmas, np.mean(np.reshape(np.abs(cond_mean_err_q), (num_sigmas, -1)), axis=1), label='q')
plt.loglog(sigmas, np.mean(np.reshape(np.abs(cond_mean_err_e), (num_sigmas, -1)), axis=1), label='e')
plt.loglog(sigmas, np.mean(np.reshape(np.abs(cond_mean_err_CK), (num_sigmas, -1)), axis=1), label='CK')
plt.loglog(sigmas, np.mean(np.reshape(np.abs(cond_mean_err_6d), (num_sigmas, -1)), axis=1), label='6d')
plt.legend()

plt.subplot(3, 1, 2)
plt.loglog(sigmas, np.mean(np.reshape(sample_var_err_q, (num_sigmas, -1)), axis=1), label='q')
plt.loglog(sigmas, np.mean(np.reshape(sample_var_err_e, (num_sigmas, -1)), axis=1), label='e')
plt.loglog(sigmas, np.mean(np.reshape(sample_var_err_CK, (num_sigmas, -1)), axis=1), label='CK')
plt.loglog(sigmas, np.mean(np.reshape(sample_var_err_6d, (num_sigmas, -1)), axis=1), label='6d')
plt.legend()

plt.subplot(3, 1, 3)
plt.loglog(sigmas, np.mean(fit_times_q, axis=1), label='q')
plt.loglog(sigmas, np.mean(fit_times_e, axis=1), label='e')
plt.loglog(sigmas, np.mean(fit_times_CK, axis=1), label='CK')
plt.loglog(sigmas, np.mean(fit_times_6d, axis=1), label='6d')
plt.legend()

plt.tight_layout()
plt.show()
