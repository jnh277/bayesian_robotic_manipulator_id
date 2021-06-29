import numpy as np
import pystan
import matplotlib.pyplot as plt



model = pystan.StanModel(file='stan/constraint.stan')


stan_data = {
    'N':1
}

fit = model.sampling(data=stan_data)

traces = fit.extract()

y1 = traces['y1']
y2 = traces['y2']
y3 = traces['y3']
y = traces['y']
R = traces['R']
inertia = traces['Inertia']

t1 = y1 + y2 - y3
t2 = y1 + y3 - y2
t3 = y2 + y3 - y1

plt.subplot(2,3,1)
plt.hist(y1)
plt.xlabel('y1')

plt.subplot(2,3,2)
plt.hist(y2)
plt.xlabel('y2')

plt.subplot(2,3,3)
plt.hist(y3)
plt.xlabel('y3')

plt.subplot(2,3,4)
plt.hist(t1)
plt.xlabel('y1 + y2 - y3')

plt.subplot(2,3,5)
plt.hist(t2)
plt.xlabel('y1 + y3 - y2')

plt.subplot(2,3,6)
plt.hist(t3)
plt.xlabel('y2 + y3 - y1')

plt.tight_layout()
plt.show()
