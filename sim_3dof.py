import numpy as np
import matplotlib.pyplot as plt
import stan
from numpy import cos, sin
from mpl_toolkits import mplot3d
from itertools import product, combinations


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



def T_func(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    T_all = [np.array([
        [cos(q1), -sin(q1), 0, 0],
        [sin(q1), cos(q1), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]), np.array([
        [cos(q1) * cos(q2), -sin(q2) * cos(q1), sin(q1), 0],
        [sin(q1) * cos(q2), -sin(q1) * sin(q2), -cos(q1), 0],
        [sin(q2), cos(q2), 0, 0],
        [0, 0, 0, 1]]), np.array([
        [-sin(q2) * sin(q3) * cos(q1) + cos(q1) * cos(q2) * cos(q3),
         -sin(q2) * cos(q1) * cos(q3) - sin(q3) * cos(q1) * cos(q2), sin(q1), a1 * cos(q1) * cos(q2)],
        [-sin(q1) * sin(q2) * sin(q3) + sin(q1) * cos(q2) * cos(q3),
         -sin(q1) * sin(q2) * cos(q3) - sin(q1) * sin(q3) * cos(q2), -cos(q1), a1 * sin(q1) * cos(q2)],
        [sin(q2) * cos(q3) + sin(q3) * cos(q2), -sin(q2) * sin(q3) + cos(q2) * cos(q3), 0, a1 * sin(q2)],
        [0, 0, 0, 1]])]
    return T_all

def forward_kin(q):
    p_l = [np.array([0, 0, d0, 0]),
           np.array([a1, 0, 0, 0]),
           np.array([a2, 0, 0, 0])]
    T = T_func(q)
    p = [T[0] @ p_l[0]]
    for i in range(1,dof):
        p.append(T[i] @ p_l[i] + p[i-1])
    return p, T

q1 = np.linspace(np.pi/4, np.pi, 2)
q2 = np.linspace(np.pi/4, np.pi/2, 2)
q3 = np.linspace(-np.pi/4, -np.pi/2, 2)
for i in range(len(q2)):
# i = 2
    q = np.array([q1[i], q2[i],q3[i]])
    p, T = forward_kin(q)


    lx = np.array([0, p[0][0], p[1][0], p[2][0]])
    ly = np.array([0, p[0][1], p[1][1], p[2][1]])
    lz = np.array([0, p[0][2], p[1][2], p[2][2]])

    fig = plt.figure()
    r = [-0.2, 0.2]
    ax = plt.axes(projection='3d')
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            s = T[0][:3, :3] @ s
            e = T[0][:3, :3] @ e
            ax.plot3D(*zip(s, e), color="k")
    ax.plot(lx, ly, lz, linewidth=2, color='k')
    ax.scatter(lx,ly,lz)
    plt.xlim([-1.3, 1.3])
    plt.ylim([-1.3, 1.3])
    ax.set_zlim(-1.3, 1.3)
    plt.show()


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



def gravity_term(parms, q):
#
    g_out = [0]*3
#
    x0 = cos(q[1])
    x1 = 9.81*x0
    x2 = cos(q[2])
    x3 = sin(q[1])
    x4 = 9.81*x3
    x5 = -x4
    x6 = sin(q[2])
    x7 = x1*x2 + x5*x6
    x8 = -parms[30]*x7
    x9 = x1*x6 + x2*x4
    x10 = parms[30]*x9
    x11 = parms[28]*x7 - parms[29]*x9
#
    g_out[0] = x0*(parms[19]*x4 + x10*x2 + x6*x8) + x3*(-parms[19]*x1 - x10*x6 + x2*x8)
    g_out[1] = a1*(parms[31]*x2*x7 + parms[31]*x6*x9) + parms[17]*x1 + parms[18]*x5 + x11
    g_out[2] = x11
    return np.reshape(np.array(g_out),(dof,1))

def mass_matrix(parms, q):
#
    M_out = [0]*9
#
    x0 = cos(q[1])
    x1 = sin(q[2])
    x2 = sin(q[1])
    x3 = cos(q[2])
    x4 = x0*x1 + x2*x3
    x5 = -a1*x0
    x6 = x0*x3 - x1*x2
    x7 = parms[22]*x4 + parms[23]*x6 + parms[29]*x5
    x8 = parms[23]*x4 + parms[25]*x6 - parms[28]*x5
    x9 = parms[24]*x4 + parms[26]*x6
    x10 = a1*(parms[30]*x1*x6 - parms[30]*x3*x4) + parms[13]*x2 + parms[15]*x0 + x9
    x11 = a1*x1
    x12 = -parms[29]
    x13 = a1*x3
    x14 = parms[27] + parms[28]*x13 + x11*x12
#
    M_out[0] = parms[5] + x0*(-a1*(-parms[28]*x6 + parms[29]*x4 + parms[31]*x5) + parms[12]*x2 + parms[14]*x0 + x1*x7 + x3*x8) + x2*(parms[11]*x2 + parms[12]*x0 - x1*x8 + x3*x7)
    M_out[1] = x10
    M_out[2] = x9
    M_out[3] = x10
    M_out[4] = a1*(x1*(parms[31]*x11 + x12) + x3*(parms[28] + parms[31]*x13)) + parms[16] + x14
    M_out[5] = x14
    M_out[6] = x9
    M_out[7] = x14
    M_out[8] = parms[27]
#
    return np.reshape(np.array(M_out), (dof,dof))

def corriolis_term(parms, q, dq):
#
    c_out = [0]*3
#
    x0 = sin(q[1])
    x1 = dq[0]*x0
    x2 = dq[1]*x1
    x3 = -x2
    x4 = a1*(x2 - x3)
    x5 = dq[1] + dq[2]
    x6 = cos(q[2])
    x7 = cos(q[1])
    x8 = dq[0]*x7
    x9 = sin(q[2])
    x10 = -x9
    x11 = x1*x10 + x6*x8
    x12 = x1*x6 + x8*x9
    x13 = parms[22]*x12 + parms[23]*x11 + parms[24]*x5
    x14 = -x12
    x15 = parms[24]*x12 + parms[26]*x11 + parms[27]*x5
    x16 = dq[1]*x8
    x17 = dq[2]*x11 + x16*x6 + x3*x9
    x18 = dq[2]*x14 - x16*x9 + x3*x6
    x19 = a1*(-dq[1]**2 - x8**2)
    x20 = a1*x1*x8
    x21 = x19*x6 + x20*x9
    x22 = parms[23]*x17 + parms[25]*x18 - parms[28]*x4 + parms[30]*x21 + x13*x5 + x14*x15
    x23 = -x12**2
    x24 = -x11**2
    x25 = x11*x5
    x26 = x12*x5
    x27 = x10*x19 + x20*x6
    x28 = parms[23]*x12 + parms[25]*x11 + parms[26]*x5
    x29 = parms[22]*x17 + parms[23]*x18 + parms[29]*x4 - parms[30]*x27 + x11*x15 - x28*x5
    x30 = dq[1]*parms[13] + parms[11]*x1 + parms[12]*x8
    x31 = dq[1]*parms[16] + parms[13]*x1 + parms[15]*x8
    x32 = dq[1]*parms[15] + parms[12]*x1 + parms[14]*x8
    x33 = x11*x12
    x34 = -x5**2
    x35 = parms[24]*x17 + parms[26]*x18 + parms[28]*x27 - parms[29]*x21 - x11*x13 + x12*x28
#
    c_out[0] = x0*(-dq[1]*x32 + parms[11]*x16 + parms[12]*x3 + x10*x22 + x29*x6 + x31*x8) + x7*(-a1*(parms[28]*(-x18 + x26) + parms[29]*(x17 + x25) + parms[30]*(x23 + x24) + parms[31]*x4) + dq[1]*x30 + parms[12]*x16 + parms[14]*x3 - x1*x31 + x22*x6 + x29*x9)
    c_out[1] = a1*(x6*(parms[28]*x33 + parms[29]*(x23 + x34) + parms[30]*(-x17 + x25) + parms[31]*x27) + x9*(parms[28]*(x24 + x34) + parms[29]*x33 + parms[30]*(x18 + x26) + parms[31]*x21)) + parms[13]*x16 + parms[15]*x3 + x1*x32 - x30*x8 + x35
    c_out[2] = x35
#
    return np.reshape(np.array(c_out),(dof,1))

def dynamics(params, q, dq, tau):
    c = corriolis_term(params, q, dq)
    M = mass_matrix(params, q)
    g = gravity_term(params, q)
    d = np.reshape(dq * params[10::11], (dof, 1))
    ddq = np.linalg.solve(M,(np.reshape(tau, (dof,1)) - c - g - d))

    return ddq

params = to_param_vector(I_all, m_all, r_all, fv_all)

def inverse_dynamics(parms, q, dq, ddq):
#
    tau_out = [0]*3
#
    x0 = sin(q[1])
    x1 = dq[0]*x0
    x2 = dq[1]*x1
    x3 = cos(q[1])
    x4 = ddq[0]*x3 - x2
    x5 = cos(q[2])
    x6 = sin(q[2])
    x7 = dq[0]*x3
    x8 = x1*x5 + x6*x7
    x9 = -x8
    x10 = ddq[0]*x0 + dq[1]*x7
    x11 = dq[2]*x9 - x10*x6 + x4*x5
    x12 = 9.81*x3
    x13 = a1*(ddq[1] + x1*x7) + x12
    x14 = 9.81*x0
    x15 = a1*(-dq[1]**2 - x7**2) + x14
    x16 = -x6
    x17 = x13*x5 + x15*x16
    x18 = x1*x16 + x5*x7
    x19 = dq[2]*x18 + x10*x5 + x4*x6
    x20 = ddq[1] + ddq[2]
    x21 = dq[1] + dq[2]
    x22 = parms[23]*x8 + parms[25]*x18 + parms[26]*x21
    x23 = a1*(x2 - x4)
    x24 = parms[24]*x8 + parms[26]*x18 + parms[27]*x21
    x25 = parms[22]*x19 + parms[23]*x11 + parms[24]*x20 + parms[29]*x23 - parms[30]*x17 + x18*x24 - x21*x22
    x26 = parms[22]*x8 + parms[23]*x18 + parms[24]*x21
    x27 = x13*x6 + x15*x5
    x28 = parms[23]*x19 + parms[25]*x11 + parms[26]*x20 - parms[28]*x23 + parms[30]*x27 + x21*x26 + x24*x9
    x29 = dq[1]*parms[15] + parms[12]*x1 + parms[14]*x7
    x30 = dq[1]*parms[16] + parms[13]*x1 + parms[15]*x7
    x31 = x18*x21
    x32 = -x18**2
    x33 = -x8**2
    x34 = x21*x8
    x35 = dq[1]*parms[13] + parms[11]*x1 + parms[12]*x7
    x36 = parms[24]*x19 + parms[26]*x11 + parms[27]*x20 + parms[28]*x17 - parms[29]*x27 - x18*x26 + x22*x8
    x37 = -x21**2
    x38 = x18*x8
#
    tau_out[0] = ddq[0]*parms[5] + dq[0]*parms[10] + x0*(ddq[1]*parms[13] - dq[1]*x29 + parms[11]*x10 + parms[12]*x4 - parms[19]*x12 + x16*x28 + x25*x5 + x30*x7) + x3*(-a1*(parms[28]*(-x11 + x34) + parms[29]*(x19 + x31) + parms[30]*(x32 + x33) + parms[31]*x23) + ddq[1]*parms[15] + dq[1]*x35 + parms[12]*x10 + parms[14]*x4 + parms[19]*x14 - x1*x30 + x25*x6 + x28*x5)
    tau_out[1] = a1*(x5*(parms[28]*(x20 + x38) + parms[29]*(x33 + x37) + parms[30]*(-x19 + x31) + parms[31]*x17) + x6*(parms[28]*(x32 + x37) + parms[29]*(-x20 + x38) + parms[30]*(x11 + x34) + parms[31]*x27)) + ddq[1]*parms[16] + dq[1]*parms[21] + parms[13]*x10 + parms[15]*x4 + parms[17]*x12 - parms[18]*x14 + x1*x29 - x35*x7 + x36
    tau_out[2] = dq[2]*parms[32] + x36
#
    return np.array(tau_out)

dt = 0.01
Tsim = 2
T = np.round(Tsim/dt).astype(int)

q = np.zeros((3, T+1))
# q[0,0] = -np.pi/2
dq = np.zeros((3, T+1))
ddq = np.zeros((3, T))
tau = np.zeros((3, T+1))
tau[0, :] = 0.5*np.ones((T+1,))
tau[1, :] = 1.*np.ones((T+1,))
tau[2, :] = 0.5*np.ones((T+1,))


for t in range(T):
    ddq[:, [t]] = dynamics(params, q[:,t], dq[:,t], tau[:,t])
    dq[:, [t+1]] = dq[:, [t]] + dt * ddq[:, [t]]
    q[:, [t+1]] = q[:, [t]] + dt * dq[:, [t]]

    # binary input switching
    if np.random.rand() > 0.95:
        tau[0, t+1] = -tau[0, t]
    else:
        tau[0, t+1] = tau[0, t]
    if np.random.rand() > 0.95:
        tau[1, t+1] = -tau[1, t]
    else:
        tau[1, t+1] = tau[1, t]
    if np.random.rand() > 0.95:
        tau[2, t+1] = -tau[1, t]
    else:
        tau[2, t+1] = tau[1, t]

tau = tau[:, :-1]
q = q[:, :-1]
dq = dq[:, :-1]

plt.subplot(3,1,1)
plt.plot(np.arange(T)*dt,q[0,:])
plt.subplot(3,1,2)
plt.plot(np.arange(T)*dt,q[1,:])
plt.subplot(3,1,3)
plt.plot(np.arange(T)*dt,q[2,:])
plt.show()

tau_hat = np.zeros((3,T))
for t in range(T):
    tau_hat[:, t] = inverse_dynamics(params, q[:, t], dq[:, t], ddq[:, t])

err = tau_hat - tau
mae = np.mean(np.abs(err))
print(mae)



r = 1e-2
tau_m = tau + np.random.normal(0,r,tau.shape)

stan_data = {
    'dof':3,
    'N':int(T),
    'q':q,
    'dq':dq,
    'ddq':ddq,
    'tau':tau_m,
    'a1':a1,
    'd0':d0
}
r_com = np.vstack((r_1,r_2,r_3))
def init_function():
    output = dict(m=[m_1*np.random.uniform(0.8,1.2),m_2*np.random.uniform(0.8,1.2),m_2 * np.random.uniform(0.8, 1.2)],
                  r_com=r_com*np.random.uniform(0.8,1.2,r_com.shape))
    return output

init = [init_function(),init_function(),init_function(),init_function()]


f = open('stan/robot_3dof_auto.stan', 'r')
model_code = f.read()
posterior = stan.build(model_code, data=stan_data)
traces = posterior.sample(init=init,num_samples=2000, num_warmup=8000, num_chains=4)

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
        dig=int(param[2])-1
        if param[3] == 'x':
            param_dict[param] = traces['l'][dig][0]
        if param[3] == 'y':
            param_dict[param] = traces['l'][dig][1]
        if param[3] == 'z':
            param_dict[param] = traces['l'][dig][2]
    elif param[0:2] == "L_":
        dig=int(param[2])-1
        name=param[0:1]+param[3:]
        param_dict[param] = traces[name][dig]
    elif param[0:2] == "m_":
        dig=int(param[-1])-1
        param_dict[param] = traces['m'][dig]
    elif param[0] == "f":
        name=param[0:2]
        dig=int(param[-1])-1
        param_dict[param] = traces[name][dig]
    elif param[0] == "Ia":
        name=param[0:2]
        dig=int(param[-1])-1
        param_dict[param] = traces[name][dig]
    else:
        print("error occured")
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

num_lumped = len(lumped_params)
num_plots = num_lumped // 9 + 1
for i in range(num_plots):
    for j in range(min(9, num_lumped - 9 * i)):
        plt.subplot(3, 3, j + 1)
        plt.hist(lumped_param_dict[lumped_params[j + 9 * i]].flatten(), bins=30, density=True)
        plt.axvline(true_lumped_param_dict[lumped_params[j + 9 * i]], linestyle='--', linewidth=2, color='k')
        plt.xlabel(lumped_params[j + 9 * i])
    plt.tight_layout()
    plt.show()


