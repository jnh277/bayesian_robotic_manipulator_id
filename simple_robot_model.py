import numpy as np
import matplotlib.pyplot as plt
import stan

a1 = 0.8        # (m) length of first link
a2 = 0.4        # (m) length of second link

I_1 = np.diag([0.05, 0.1, 0.8])
I_2 = I_1 / 2.
m_1 = 0.75
m_2 = 0.3
r_1 = np.array([a1/2, 0, 0])
r_2 = np.array([a2/2, 0, 0])
fv_1 = 0.2
fv_2 = 0.1


# rbt kinematics

def T1_func(q):
    q1 = q[0]
    T1 = np.array([[np.cos(q1), -np.sin(q1),  0, 0],
                    [      0,        0, -1, 0],
                    [np.sin(q1),  np.cos(q1),  0, 0],
                    [      0,        0,  0, 1]])
    return T1

def T2_func(q):
    q1 = q[0]
    q2 = q[1]
    T2 = np.array([
        [-np.sin(q1)*np.sin(q2) + np.cos(q1)*np.cos(q2), -np.sin(q1)*np.cos(q2) - np.sin(q2)*np.cos(q1),  0, a1*np.cos(q1)],
        [                                 0,                                  0, -1,          0],
        [ np.sin(q1)*np.cos(q2) + np.sin(q2)*np.cos(q1), -np.sin(q1)*np.sin(q2) + np.cos(q1)*np.cos(q2),  0, a1*np.sin(q1)],
        [                                 0,                                  0,  0,          1]])
    return T2

def forward_kin(q):
    p1_l = np.array([a1, 0, 0, 0])
    p2_l = np.array([a2, 0, 0, 0])
    T1 = T1_func(q)
    T2 = T2_func(q)
    p1 = T1 @ p1_l
    p2 = T2 @ p2_l + p1
    return p1, p2


q1 = np.linspace(0, -np.pi/2, 3)
q2 = np.linspace(0, 3*np.pi, 3)
p1s = []
for i in range(len(q1)):
# i = 2
    q = np.array([q1[i],q2[i]])
    p1, p2 = forward_kin(q)
    p1s.append(p1)

    lx = np.array([0, p1[0], p2[0]])
    lz = np.array([0, p1[2], p2[2]])

    plt.plot(lx, lz, 'ok-')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.show()


# unknown parameters
# we want to estiamte inertia about com, mass, and position of cetner of mass from joint coordinate system

# the model takes in parameters
# [L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1x, l_1y, l_1z, m_1, fv_1, L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2x, l_2y, l_2z, m_2, fv_2]

def L_funcof_I(I_1,r_1,m_1, I_2, r_2, m_2):
    """"
    I: inertia tensor about COM
    r: position of center of mass with respect to link coordinate system
    m: mass of link
    """
    I_1xx = I_1[0, 0]
    I_1yy = I_1[1, 1]
    I_1zz = I_1[2, 2]
    I_1xy = I_1[0, 1]
    I_1xz = I_1[0, 2]
    I_1yz = I_1[1, 2]
    r_1x = r_1[0]
    r_1y = r_1[1]
    r_1z = r_1[2]
    I_2xx = I_2[0, 0]
    I_2yy = I_2[1, 1]
    I_2zz = I_2[2, 2]
    I_2xy = I_2[0, 1]
    I_2xz = I_2[0, 2]
    I_2yz = I_2[1, 2]
    r_2x = r_2[0]
    r_2y = r_2[1]
    r_2z = r_2[2]
    L_1 = np.array([
        [I_1xx + m_1*r_1y**2 + m_1*r_1z**2,             I_1xy - m_1*r_1x*r_1y,             I_1xz - m_1*r_1x*r_1z],
        [            I_1xy - m_1*r_1x*r_1y, I_1yy + m_1*r_1x**2 + m_1*r_1z**2,             I_1yz - m_1*r_1y*r_1z],
        [            I_1xz - m_1*r_1x*r_1z,             I_1yz - m_1*r_1y*r_1z, I_1zz + m_1*r_1x**2 + m_1*r_1y**2]])
    L_2 = np.array([
        [I_2xx + m_2*r_2y**2 + m_2*r_2z**2,             I_2xy - m_2*r_2x*r_2y,             I_2xz - m_2*r_2x*r_2z],
        [            I_2xy - m_2*r_2x*r_2y, I_2yy + m_2*r_2x**2 + m_2*r_2z**2,             I_2yz - m_2*r_2y*r_2z],
        [            I_2xz - m_2*r_2x*r_2z,             I_2yz - m_2*r_2y*r_2z, I_2zz + m_2*r_2x**2 + m_2*r_2y**2]])
    return L_1, L_2

def l_funcof_r(r_1, m_1, r_2, m_2):
    l_1 = r_1 * m_1
    l_2 = r_2 * m_2
    return l_1, l_2


def to_param_vector(I_1, r_1, m_1, fv_1, I_2, r_2, m_2, fv_2):
    L_1, L_2 = L_funcof_I(I_1,r_1,m_1, I_2, r_2, m_2)
    l_1, l_2 = l_funcof_r(r_1, m_1, r_2, m_2)
    tri_inds = np.triu_indices(L_1.shape[0])
    L_1[tri_inds]
    params = np.hstack([L_1[tri_inds], l_1, m_1, fv_1, L_2[tri_inds], l_2, m_2, fv_2])
    return params
# dynamics

def gravity_term(parms, q):
    g_out = [0]*2
    x0 = 9.81*np.cos(q[0])
    x1 = np.cos(q[1])
    x2 = 9.81*np.sin(q[0])
    x3 = -x2
    x4 = np.sin(q[1])
    x5 = x0*x1 + x3*x4
    x6 = x0*x4 + x1*x2
    x7 = parms[17]*x5 - parms[18]*x6
    #
    g_out[0] = a1*(parms[20]*x1*x5 + parms[20]*x4*x6) + parms[6]*x0 + parms[7]*x3 + x7
    g_out[1] = x7
    #
    return np.reshape(np.array(g_out),(2,1))


def mass_matrix(parms, q, dq):
    #
    M_out = [0]*4
    #
    x0 = np.sin(q[1])
    x1 = a1*x0
    x2 = -parms[18]
    x3 = np.cos(q[1])
    x4 = a1*x3
    x5 = parms[16] + parms[17]*x4 + x1*x2
    #
    M_out[0] = a1*(x0*(parms[20]*x1 + x2) + x3*(parms[17] + parms[20]*x4)) + parms[5] + x5
    M_out[1] = x5
    M_out[2] = x5
    M_out[3] = parms[16]
    #
    return np.reshape(np.array(M_out),(2,2))

def corriolis_term(parms, q, dq):
    #
    c_out = [0]*2
    #
    x0 = -(dq[0] + dq[1])**2
    x1 = np.cos(q[1])
    x2 = -a1*dq[0]**2
    x3 = x1*x2
    x4 = np.sin(q[1])
    x5 = -x2*x4
    x6 = parms[17]*x5 - parms[18]*x3
    #
    c_out[0] = a1*(x1*(parms[18]*x0 + parms[20]*x5) + x4*(parms[17]*x0 + parms[20]*x3)) + x6
    c_out[1] = x6
    #
    return np.reshape(np.array(c_out), (2,1))

def dynamics(params, q, dq, tau):
    dof = len(q)
    c = corriolis_term(params, q, dq)
    M = mass_matrix(params, q, dq)
    g = gravity_term(params, q)
    d = np.array([[dq[0] * params[10]],
                  [dq[1] * params[21]]])
    ddq = np.linalg.solve(M,(np.reshape(tau, (dof,1)) - c - g - d))

    return ddq


def inverse_dynamics(parms, q, dq, ddq):
    #
    tau_out = [0]*2
    #
    x0 = np.cos(q[1])
    x1 = ddq[0] + ddq[1]
    x2 = -(dq[0] + dq[1])**2
    x3 = 9.81*np.sin(q[0])
    x4 = -a1*dq[0]**2 + x3
    x5 = np.sin(q[1])
    x6 = 9.81*np.cos(q[0])
    x7 = a1*ddq[0] + x6
    x8 = x0*x7 - x4*x5
    x9 = x0*x4 + x5*x7
    x10 = parms[16]*x1 + parms[17]*x8 - parms[18]*x9
    #
    tau_out[0] = a1*(x0*(parms[17]*x1 + parms[18]*x2 + parms[20]*x8) + x5*(parms[17]*x2 - parms[18]*x1 + parms[20]*x9)) + ddq[0]*parms[5] + dq[0]*parms[10] + parms[6]*x6 - parms[7]*x3 + x10
    tau_out[1] = dq[1]*parms[21] + x10
    #
    return np.array(tau_out)

params = to_param_vector(I_1, r_1, m_1, fv_1, I_2, r_2, m_2, fv_2)


dt = 0.01
Tsim = 6
T = np.round(Tsim/dt).astype(int)

q = np.zeros((2, T+1))
q[0,0] = -np.pi/2
dq = np.zeros((2, T+1))
ddq = np.zeros((2, T))
tau = np.zeros((2, T+1))
tau[0, :] = 1.*np.ones((T+1,))
tau[1, :] = 0.5*np.ones((T+1,))




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

tau = tau[:, :-1]
q = q[:, :-1]
dq = dq[:, :-1]

plt.subplot(2,1,1)
plt.plot(np.arange(T)*dt,q[0,:])
plt.subplot(2,1,2)
plt.plot(np.arange(T)*dt,q[1,:])
plt.show()

tau_hat = np.zeros((2,T))
for t in range(T):
    tau_hat[:,t] = inverse_dynamics(params, q[:, t], dq[:, t], ddq[:, t])

err = tau_hat - tau
mae = np.mean(np.abs(err))
print(mae)


plt.subplot(2,1,1)
plt.plot(np.arange(T)*dt, tau[0, :])
plt.plot(np.arange(T)*dt, tau_hat[0, :],'--')
plt.plot(np.arange(T)*dt, tau[0, :] - tau_hat[0, :])

plt.subplot(2,1,2)
plt.plot(np.arange(T)*dt, tau[1, :])
plt.plot(np.arange(T)*dt, tau_hat[1, :],'--')
plt.plot(np.arange(T)*dt, tau[1, :] - tau_hat[1, :])
plt.show()

r = 1e-2
tau_m = tau + np.random.normal(0,r,tau.shape)

stan_data = {
    'N':int(T),
    'q':q,
    'dq':dq,
    'ddq':ddq,
    'tau':tau_m,
    'a1':a1
}

def init_function():
    output = dict(m_1=m_1*np.random.uniform(0.8,1.2),
                  m_2=m_2*np.random.uniform(0.8,1.2),
                  I_1=I_1*np.random.uniform(0.8,1.2,I_1.shape),
                  I_2=I_2*np.random.uniform(0.8,1.2,I_2.shape),
                  r_1=r_1*np.random.uniform(0.8,1.2,r_1.shape),
                  r_2=r_2*np.random.uniform(0.8,1.2,r_2.shape))
    return output

init = [init_function(),init_function(),init_function(),init_function()]


f = open('stan/simple_robot_v2.stan', 'r')
model_code = f.read()
posterior = stan.build(model_code, data=stan_data)
traces = posterior.sample(init=init,num_samples=2000, num_warmup=8000, num_chains=4)

I_1_hat = traces['I_1']
I_2_hat = traces['I_2']
r_1_hat = traces['r_1']
r_2_hat = traces['r_2']
m_1_hat = traces['m_1']
m_2_hat = traces['m_2']
r_hat = traces['r']
fv_1_hat = traces['fv_1']
fv_2_hat = traces['fv_2']

plt.hist(r_hat[0], bins=30)
# plt.axvline(m_1, linestyle='--', linewidth=2, color='k')
plt.show()

plt.subplot(2,2,1)
plt.hist(m_1_hat[0], bins=30)
plt.axvline(m_1, linestyle='--', linewidth=2, color='k')
plt.title('m_1')
plt.xlim([0, m_1*5])

plt.subplot(2,2,2)
plt.hist(m_2_hat[0], bins=30)
plt.axvline(m_2, linestyle='--', linewidth=2, color='k')
plt.title('m_2')
plt.xlim([0, m_2*5])

plt.subplot(2,2,3)
plt.hist(fv_1_hat[0], bins=30)
plt.axvline(fv_1, linestyle='--', linewidth=2, color='k')
plt.title('fv_1')
# plt.xlim([0, fv_1*5])

plt.subplot(2,2,4)
plt.hist(fv_2_hat[0], bins=30)
plt.axvline(fv_2, linestyle='--', linewidth=2, color='k')
plt.title('fv_2')
# plt.xlim([0, fv_2*5])

plt.tight_layout()
plt.show()


for i in range(9):
    plt.subplot(3,3,i+1)
    plt.hist(I_1_hat[i //3, i % 3], bins=30, range=[0, I_1[i //3, i % 3] * 5])
    plt.axvline(I_1[i //3, i % 3], linestyle='--', linewidth=2, color='k')
    if i==1:
        plt.title('I_1')
plt.tight_layout()
plt.show()

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.hist(I_2_hat[i //3, i % 3], bins=30, range=[0, I_2[i //3, i % 3] * 5])
    plt.axvline(I_2[i //3, i % 3], linestyle='--', linewidth=2, color='k')
    if i==1:
        plt.title('I_2')
plt.tight_layout()
plt.show()

for i in range(3):
    plt.subplot(2,3,i+1)
    plt.hist(r_1_hat[i], bins=30)
    plt.axvline(r_1[i], linestyle='--', linewidth=2, color='k')
    if i==0:
        plt.ylabel('r_1')
    plt.subplot(2,3,i+1+3)
    plt.hist(r_2_hat[i], bins=30)
    plt.axvline(r_2[i], linestyle='--', linewidth=2, color='k')
    if i==0:
        plt.ylabel('r_2')

plt.tight_layout()
plt.show()

# parameters that actually affect the model
# [L_1zz + 16*m_2/25],
# [   l_1x + 4*m_2/5],
# [             l_1y],
# [             fv_1],
# [            L_2zz],
# [             l_2x],
# [             l_2y],
# [             fv_2]])

# which gives us the following observable root parameters (not quite how this works)
# I_1zz,
# r_1x, r_1y
# m_1, m_2
# fv_1, fv_2
# r_2x, r_2y,
# I_2zz

L_1zz_hat = traces['L_1zz']
L_2zz_hat = traces['L_2zz']
l_1_hat = traces['l_1']
l_2_hat = traces['l_2']


L_1, L_2 = L_funcof_I(I_1, r_1, m_1, I_2, r_2, m_2)
l_1, l_2 = l_funcof_r(r_1, m_1, r_2, m_2)

plt.subplot(3,3,1)
plt.hist(L_1zz_hat[0,0] + 16/25 * m_2_hat[0], bins=30, density=True)
plt.axvline(L_1[2,2] + 16/25 * m_2, linestyle='--', linewidth=2, color='k')
plt.xlabel('L_1zz + 16*m_2/25')

plt.subplot(3,3,2)
plt.hist(l_1_hat[0] + 4/5*m_2_hat[0], bins=30, density=True)
plt.axvline(l_1[0] + 4/5 * m_2, linestyle='--', linewidth=2, color='k')
plt.xlabel('l_1x + 4*m_2/5')
plt.title('BASE PARAMS')

plt.subplot(3,3,3)
plt.hist(l_1_hat[1], bins=30, density=True)
plt.axvline(l_1[1], linestyle='--', linewidth=2, color='k')
plt.xlabel('l_1y')

plt.subplot(3,3,4)
plt.hist(fv_1_hat[0], bins=30, density=True)
plt.axvline(fv_1, linestyle='--', linewidth=2, color='k')
plt.xlabel('fv_1')

plt.subplot(3,3,5)
plt.hist(fv_2_hat[0], bins=30, density=True)
plt.axvline(fv_2, linestyle='--', linewidth=2, color='k')
plt.xlabel('fv_2')

plt.subplot(3,3,6)
plt.hist(L_2zz_hat[0], bins=30, density=True)
plt.axvline(L_2[2,2], linestyle='--', linewidth=2, color='k')
plt.xlabel('L_2zz')

plt.subplot(3,3,7)
plt.hist(l_2_hat[0], bins=30, density=True)
plt.axvline(l_2[0], linestyle='--', linewidth=2, color='k')
plt.xlabel('l_2x')

plt.subplot(3,3,8)
plt.hist(l_2_hat[1], bins=30, density=True)
plt.axvline(l_2[1], linestyle='--', linewidth=2, color='k')
plt.xlabel('l_2y')

plt.subplot(3,3,9)
plt.hist(r_hat[0], bins=30, density=True)
plt.axvline(r, linestyle='--', linewidth=2, color='k')
plt.xlabel('noise std')

plt.tight_layout()
plt.show()

# look at tau estimates
tau_hat = traces['tau_hat']
cm_tau_hat = tau_hat.mean(axis=2)
tau_mse = ((cm_tau_hat - tau) **2).mean()
tau_err_var = np.mean((np.reshape(tau, (2,-1,1)) - tau_hat)**2)

for i in range(3):
    plt.subplot(2,3,i+1)
    plt.hist(tau_hat[0, i*100+10, :], density=True, bins=30)
    plt.axvline(tau[0, i*100+10], linestyle='--', linewidth=2, color='k')
    plt.xlabel('tau[0,'+str(i*100+10)+']')

    plt.subplot(2,3,i+1+3)
    plt.hist(tau_hat[1, i*100+10, :], density=True, bins=30)
    plt.axvline(tau[1, i*100+10], linestyle='--', linewidth=2, color='k')
    plt.xlabel('tau[1,'+str(i*100+10)+']')

plt.tight_layout()
plt.show()