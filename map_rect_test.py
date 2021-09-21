import numpy as np
import matplotlib.pyplot as plt
# import stan
from numpy import cos, sin
import tqdm
from mpl_toolkits import mplot3d
import time
from itertools import product, combinations
import json
import os
from cmdstanpy import cmdstan_path, CmdStanModel, set_cmdstan_path

set_cmdstan_path('../cmdstan')
data_file = os.path.join(cmdstan_path(), 'models', 'map_rect_test.data.json')

N = 100000
x1 = np.random.uniform(0, 1, (N,))
x2 = np.random.uniform(-1, 1, (N,))
r = 0.1
theta = 0.5

y = np.random.normal(theta*x1+x2, r)
y2 = np.random.normal(x2+theta*x1, r)

stan_data = {
    'N':N,
    'k':10,
    'x1':x1.tolist(),
    'x2':x2.tolist(),
    'y':y.tolist(),
    'grainsize':int(N // 16),
    'y2':y2.tolist(),
    'y_all':(np.vstack((y, y2)).T).tolist()
}

with open(data_file, 'w') as outfile:
    json.dump(stan_data, outfile)

stan_file = os.path.join(cmdstan_path(), 'models', 'map_rect_test.stan')
model = CmdStanModel(stan_file=stan_file)
# model = CmdStanModel(stan_file=stan_file)
time_start = time.time()
fit = model.sample(chains=2, data=data_file)
time_finish = time.time()
print('Sampling took ', time_finish - time_start, ' seconds')

traces = fit.draws_pd()
theta_hat = traces['theta']
r_hat = traces['r']

plt.hist(r_hat,bins=30)
plt.show()

plt.hist(theta_hat, bins=30)
plt.show()

stan_file = os.path.join(cmdstan_path(), 'models', 'map_rect_test2.stan')
model = CmdStanModel(stan_file=stan_file, cpp_options={"STAN_THREADS": True})
time_start = time.time()
fit = model.sample(chains=2, data=data_file)
time_finish = time.time()
print('Parrallel sampling took ', time_finish - time_start, ' seconds')

traces = fit.draws_pd()
theta_hat = traces['theta']
r_hat = traces['r']

plt.hist(r_hat,bins=30)
plt.show()

plt.hist(theta_hat, bins=30)
plt.show()