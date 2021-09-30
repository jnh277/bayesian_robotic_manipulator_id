import numpy as np
import matplotlib.pyplot as plt
import stan
from numpy import cos, sin
from mpl_toolkits import mplot3d
from itertools import product, combinations
# import pickle
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# simulation parameters
dof = 3
# first link is jsut a rotating base
d0 = 0.0        # offset upwards from base
a1 = 0.8        # (m) length of second link
a2 = 0.4        # (m) length of third link


I_1 = np.diag([0.2, 0.3, 0.4])
I_2 = np.diag([0.2, 0.3, 0.4]) * 1.5
I_3 = I_2 / 2.
m_1 = 1.0
m_2 = 0.75
m_3 = 0.3
r_1 = np.array([0, 0, 0])
r_2 = np.array([a1/2, 0, 0])
r_3 = np.array([a2/2, 0, 0])
fv_1 = 0.01
fv_2 = 0.2
fv_3 = 0.1

I_all = [I_1, I_2, I_3]
r_all = [r_1, r_2, r_3]
m_all = [m_1, m_2, m_3]
fv_all = [fv_1, fv_2, fv_3]

def L_funcof_I(I,r,m):
    """"
    I: inertia tensor about COM
    r: position of center of mass with respect to link coordinate system
    m: mass of link
    """
    I_xx = I[0, 0]
    I_yy = I[1, 1]
    I_zz = I[2, 2]
    I_xy = I[0, 1]
    I_xz = I[0, 2]
    I_yz = I[1, 2]
    r_x = r[0]
    r_y = r[1]
    r_z = r[2]
    L = np.array([
        [I_xx + m*r_y**2 + m*r_z**2,             I_xy - m*r_x*r_y,             I_xz - m*r_x*r_z],
        [            I_xy - m*r_x*r_y, I_yy + m*r_x**2 + m*r_z**2,             I_yz - m*r_y*r_z],
        [            I_xz - m*r_x*r_z,             I_yz - m*r_y*r_z, I_zz + m*r_x**2 + m*r_y**2]])
    return L

def l_funcof_r(r, m):
    l = r * m
    return l

def to_param_vector(I_all, m_all, r_all, fv_all):
    # param vector needs to be
    # [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1x, l_1y, l_1z, m_1, fv_1, L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz,
     # l_2x, l_2y, l_2z, m_2, fv_2, L_3xx, L_3xy, L_3xz, L_3yy, L_3yz, L_3zz, l_3x, l_3y, l_3z, m_3, fv_3]
    param_list = []
    tri_inds = np.triu_indices(3)
    for i, group in enumerate(zip(I_all, r_all, m_all , fv_all)):
        L = L_funcof_I(group[0], group[1], group[2])
        l = l_funcof_r(group[1], group[2])
        param_list.extend([L[tri_inds], l[0], l[1], l[2], group[2], group[3]])
    return np.hstack(param_list)


params = to_param_vector(I_all, m_all, r_all, fv_all)

## load results
traces = pd.read_csv('results/3dof_fit.csv')


## plotting lumped params
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
            param_dict[param] = traces[param[0:3]+'.1'].to_numpy()
        if param[3] == 'y':
            param_dict[param] = traces[param[0:3]+'.2'].to_numpy()
        if param[3] == 'z':
            param_dict[param] = traces[param[0:3]+'.3'].to_numpy()
    else:
        param_dict[param] = traces[param].to_numpy()

expressions_list = lumped_params.copy()
for i in range(len(expressions_list)):
    for p in param_list:
        expressions_list[i] = expressions_list[i].replace(p, "param_dict[\"" + p + "\"]")

lumped_param_dict = dict()
for i, expression in enumerate(expressions_list):
    exec("lumped_param_dict[lumped_params[i]]=" + expression)

## get the true values
param_names = []
exec("param_names = [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1x, l_1y, l_1z, m_1, fv_1, L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2x, l_2y, l_2z, m_2, fv_2, L_3xx, L_3xy, L_3xz, L_3yy, L_3yz, L_3zz, l_3x, l_3y, l_3z, m_3, fv_3]".replace(
        ", ", "\", \"").replace("[", "[\"").replace("]", "\"]"))

true_param_dict = dict()
for i, p in enumerate(param_names):
    true_param_dict[p] = params[i]

expressions_list = lumped_params.copy()
for i in range(len(expressions_list)):
    for p in param_list:
        expressions_list[i] = expressions_list[i].replace(p, "true_param_dict[\"" + p + "\"]")

true_lumped_param_dict = dict()
for i, expression in enumerate(expressions_list):
    exec("true_lumped_param_dict[lumped_params[i]]=" + expression)

# plot everything
import types
pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
def bottom_offset(self, bboxes, bboxes2):
    bottom = self.axes.bbox.ymin
    self.offsetText.set(va="top", ha="left")
    oy = bottom - pad * self.figure.dpi / 72.0
    self.offsetText.set_position((1.1, oy))

num_lumped = len(lumped_params)
num_plots = num_lumped // 9
for i in range(num_plots):
    fig, ax = plt.subplots(3,3)
    for j in range(min(9, num_lumped - 9 * i)):
        # plt.subplot(3, 3, j + 1)
        ax[j // 3, j % 3].hist(lumped_param_dict[lumped_params[j + 9 * i]].flatten(), bins=30, density=True,label='posterior estimate')
        ax[j // 3, j % 3].axvline(true_lumped_param_dict[lumped_params[j + 9 * i]], linestyle='--', linewidth=2, color='k',label='true')
        ax[j // 3, j % 3].set_xlabel(lumped_params[j + 9 * i])
        ax[j // 3, j % 3].set_yticks([])
        ax[j // 3, j % 3].ticklabel_format(useOffset=False,style='sci',scilimits=(-1,1))
        ax[j // 3, j % 3].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[j // 3, j % 3].xaxis)
    # plt.legend(handles=legend_elements, loc='lower center')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0,-0.075, 1, 0.1),borderaxespad=1,ncol=2,loc='lower center')
    fig.tight_layout()
    fig.savefig('figures/lumped_params_'+str(i)+'.png',bbox_inches='tight')
    fig.show()


