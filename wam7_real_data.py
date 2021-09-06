import numpy as np
import matplotlib.pyplot as plt
import stan
import random

trajectory = "traj1"
data = np.load('procdata/'+trajectory+"_proc.npz")
t = data['t']
q = data['q'].T
dq = data['dq'].T
ddq = data['ddq'].T
tau = data['tau'].T

dof = q.shape[0]

N = len(t)

## use a random subset of the data
N_est = 250
inds = np.random.choice(N, N_est)

t_est = t[inds]
q_est = q[:, inds]
dq_est = dq[:, inds]
ddq_est = ddq[:, inds]
tau_est = tau[:, inds]

stan_data = {
    'dof':7,
    'N':N_est,
    'q':q_est,
    'dq':dq_est,
    'ddq':ddq_est,
    'tau':tau_est,
    'sign_dq':np.sign(dq_est)
}

# def init_function():
#     output = dict(m_1=m_1*np.random.uniform(0.8,1.2),
#                   m_2=m_2*np.random.uniform(0.8,1.2),
#                   m_3=m_2 * np.random.uniform(0.8, 1.2),
#                   r_1=r_1*np.random.uniform(0.8,1.2,r_1.shape),
#                   r_2=r_2*np.random.uniform(0.8,1.2,r_2.shape),
#                   r_3=r_3 * np.random.uniform(0.8, 1.2, r_2.shape))
#     return output



f = open('stan/wam7.stan', 'r')
model_code = f.read()
posterior = stan.build(model_code, data=stan_data)
traces = posterior.sample(num_samples=2000, num_warmup=4000, num_chains=4)

# plotting lumped params
lumped_params = ["L_1zz + L_2yy + L_3yy + 16*m_3/25",
               "fv_1",
               "L_2xx - L_2yy - 16*m_3/25",
               "L_2xy",
               "L_2xz - 4*l_3z/5",
               "L_2yz",
               "L_2zz + 16*m_3/25",
               "l_2x + 4*m_3/5",
               "l_2y",
               "fv_2",
               "L_3xx - L_3yy",
               "L_3xy",
               "L_3xz",
               "L_3yz",
               "L_3zz",
               "l_3x",
               "l_3y",
               "fv_3"]

import re

param_list = []
for param_str in lumped_params:
    split_str = re.split(" \+ | \- |\*|/", param_str)
    for sub_str in split_str:
        if not sub_str.isdigit():
            param_list.append(sub_str)

param_list = set(param_list)
param_dict = dict()

for param in param_list:
    if param[0:2] == "l_":
        if param[3] == 'x':
            param_dict[param] = traces[param[0:3]][0]
        if param[3] == 'y':
            param_dict[param] = traces[param[0:3]][1]
        if param[3] == 'z':
            param_dict[param] = traces[param[0:3]][2]
    else:
        param_dict[param] = traces[param]

expressions_list = lumped_params.copy()
for i in range(len(expressions_list)):
    for p in param_list:
        expressions_list[i] = expressions_list[i].replace(p, "param_dict[\"" + p + "\"]")

lumped_param_dict = dict()
for i, expression in enumerate(expressions_list):
    exec("lumped_param_dict[lumped_params[i]]=" + expression)

## get the true values
param_names = []
exec("param_names = [L_1xx,L_1xy,L_1xz,L_1yy,L_1yz,L_1zz,l_1x,l_1y,l_1z,m_1,Ia_1,fv_1,fc_1,fo_1,L_2xx,L_2xy,L_2xz,L_2yy,L_2yz,L_2zz,l_2x,l_2y,l_2z,m_2,Ia_2,fv_2,fc_2,fo_2,L_3xx,L_3xy,L_3xz,L_3yy,L_3yz,L_3zz,l_3x,l_3y,l_3z,m_3,Ia_3,fv_3,fc_3,fo_3,L_4xx,L_4xy,L_4xz,L_4yy,L_4yz,L_4zz,l_4x,l_4y,l_4z,m_4,Ia_4,fv_4,fc_4,fo_4,L_5xx,L_5xy,L_5xz,L_5yy,L_5yz,L_5zz,l_5x,l_5y,l_5z,m_5,Ia_5,fv_5,fc_5,fo_5,L_6xx,L_6xy,L_6xz,L_6yy,L_6yz,L_6zz,l_6x,l_6y,l_6z,m_6,Ia_6,fv_6,fc_6,fo_6,L_7xx,L_7xy,L_7xz,L_7yy,L_7yz,L_7zz,l_7x,l_7y,l_7z,m_7,Ia_7,fv_7,fc_7, fo_7]".replace(
        ", ", "\", \"").replace("[", "[\"").replace("]", "\"]"))

# true_param_dict = dict()
# for i, p in enumerate(param_names):
#     true_param_dict[p] = params[i]

# expressions_list = lumped_params.copy()
# for i in range(len(expressions_list)):
#     for p in param_list:
#         expressions_list[i] = expressions_list[i].replace(p, "true_param_dict[\"" + p + "\"]")

# true_lumped_param_dict = dict()
# for i, expression in enumerate(expressions_list):
#     exec("true_lumped_param_dict[lumped_params[i]]=" + expression)

# plot everything

num_lumped = len(lumped_params)
num_plots = num_lumped // 9 + 1
for i in range(num_plots):
    for j in range(min(9, num_lumped - 9 * i)):
        plt.subplot(3, 3, j + 1)
        plt.hist(lumped_param_dict[lumped_params[j + 9 * i]].flatten(), bins=30, density=True)
        # plt.axvline(true_lumped_param_dict[lumped_params[j + 9 * i]], linestyle='--', linewidth=2, color='k')
        plt.xlabel(lumped_params[j + 9 * i])
    plt.tight_layout()
    plt.show()


