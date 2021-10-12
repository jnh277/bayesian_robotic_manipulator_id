# Physically feasible Bayesian Robotic Manipulator Parameter Identification

This repository contains code that demonstrates physically feasible Bayesian robotic manipulator
parameter identification using Hamiltonian Monte Carlo (HMC).

## Requirements

- numpy
- matplotlib
- pystan 3
- pandas

If you would also like to regenerate the inverse dynamics code then you will need

- sympybotics
- sympy 0.7.5 (newer version don't work with sympybotics)

## Running the 3 dof simulated example

The simulated parameter identification example can be run using
```
python sim_3dof.py
```
This will use stan to sample from the posterior distribution and can take >10 hours. 
Hence presaved results have been included with the repository and can be plotted using
```
python plot_results.py
```

To regenerate the inverse dynamics code that is used in the model block of the stan model file
you can run
```
python sympy_3dof.py
```
