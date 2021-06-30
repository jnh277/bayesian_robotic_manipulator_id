import numpy as np
import pystan
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

num_tests = 100

sigma = 5e-2
inertias = np.zeros((num_tests, 3, 3))
inertias_hat = np.zeros((num_tests, 3, 3))

count = 0

for i in range(num_tests):
    # make a random postive definite symmetric matrix

    if i > 0:
        trial_matrix = np.random.uniform(-1, 1, (3, 3))
        trial_matrix = trial_matrix @ trial_matrix.T
        w, _ = np.linalg.eig(trial_matrix)
        tri_constraint_1 = w[0] + w[1] > w[2]
        tri_constraint_2 = w[0] + w[2] > w[1]
        tri_constraint_3 = w[1] + w[2] > w[0]

        while not (tri_constraint_1 and tri_constraint_2 and tri_constraint_3):
            trial_matrix = np.random.uniform(-1, 1, (3, 3))
            trial_matrix = trial_matrix @ trial_matrix.T
            w, _ = np.linalg.eig(trial_matrix)
            tri_constraint_1 = w[0] + w[1] > w[2]
            tri_constraint_2 = w[0] + w[2] > w[1]
            tri_constraint_3 = w[1] + w[2] > w[0]

            count = count + 1

        inertias[i, :, :] = trial_matrix
    else:
        inertias[i, :, :] = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]


    n = 3 # number of measurements to use per trial
    taus = np.random.uniform(-1, 1, (3, n))
    # taus[:,3]
    taus[:, 0] = [1, 0, 0]
    taus[:, 1] = [0, 1, 0]
    taus[:, 2] = [0, 0, 1]

    y = np.zeros((num_tests, 3, n))

    # for j in range(n):
    y[i, :, :] = np.linalg.solve(inertias[i, :, :], taus) + np.random.normal(0., 1e-2, (3, n))

    model_path = 'stan/inertia.pkl'
    if Path(model_path).is_file():
        model = pickle.load(open(model_path, 'rb'))
    else:
        model = pystan.StanModel(file='stan/inertia.stan')
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)


    stan_data = {
        'N': n,
        'tau': taus,
        'y': y[i, :, :]
    }

    fit = model.sampling(data=stan_data)

    traces = fit.extract()

    sigma_hat = traces['r']
    eigs_hat = traces['eigs']
    inertia_hat = traces['Inertia']



    inertias_hat[i, :, :] = inertia_hat.mean(axis=0)

    if not i % 10:   # plot every 10th estimate
        plt.subplot(3, 3, 1)
        plt.hist(inertia_hat[:, 0, 0], density=True, bins=30)
        plt.axvline(inertias[i, 0, 0], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I11')

        plt.subplot(3, 3, 2)
        plt.hist(inertia_hat[:, 0, 1], density=True, bins=30)
        plt.axvline(inertias[i, 0, 1], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I12')
        plt.title('test '+str(i))

        plt.subplot(3, 3, 3)
        plt.hist(inertia_hat[:, 0, 2], density=True, bins=30)
        plt.axvline(inertias[i, 0, 2], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I13')

        plt.subplot(3, 3, 4)
        plt.hist(inertia_hat[:, 1, 0], density=True, bins=30)
        plt.axvline(inertias[i, 1, 0], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I21')

        plt.subplot(3, 3, 5)
        plt.hist(inertia_hat[:, 1, 1], density=True, bins=30)
        plt.axvline(inertias[i, 1, 1], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I22')

        plt.subplot(3, 3, 6)
        plt.hist(inertia_hat[:, 1, 2], density=True, bins=30)
        plt.axvline(inertias[i, 1, 2], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I23')

        plt.subplot(3, 3, 7)
        plt.hist(inertia_hat[:, 2, 0], density=True, bins=30)
        plt.axvline(inertias[i, 2, 0], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I31')

        plt.subplot(3, 3, 8)
        plt.hist(inertia_hat[:, 2, 1], density=True, bins=30)
        plt.axvline(inertias[i, 2, 1], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I32')

        plt.subplot(3, 3, 9)
        plt.hist(inertia_hat[:, 2, 2], density=True, bins=30)
        plt.axvline(inertias[i, 2, 2], color='k', linestyle='--', linewidth=2.)
        plt.xlabel('I33')
        # plt.xlim((inertias[i, 2, 2]-0.5,inertias[i, 2, 2]+0.5))

        plt.tight_layout()
        plt.show()

# some error analysis
errors = inertias - inertias_hat

max_err = np.max(np.abs(errors))

plt.hist(np.reshape(errors,(-1)), density=True, bins=30)
plt.xlabel('simulation errors in estimated inertia matrix elements')
plt.show()

# plot an example of all samples satisfyig triangle constraints
tri_constraint_1 = eigs_hat[:, 0] + eigs_hat[:, 1] - eigs_hat[:, 2]
tri_constraint_2 = eigs_hat[:, 0] + eigs_hat[:, 2] - eigs_hat[:, 1]
tri_constraint_3 = eigs_hat[:, 1] + eigs_hat[:, 2] - eigs_hat[:, 0]

plt.subplot(3, 1, 1)
plt.hist(tri_constraint_1, density=True, bins=30)
plt.xlabel('e1 + e2 - e3 > 0')

plt.subplot(3, 1, 2)
plt.hist(tri_constraint_2, density=True, bins=30)
plt.xlabel('e1 + e3 - e2 > 0')

plt.subplot(3, 1, 3)
plt.hist(tri_constraint_3, density=True, bins=30)
plt.xlabel('e2 + e3 - e1 > 0')

plt.tight_layout()
plt.show()

