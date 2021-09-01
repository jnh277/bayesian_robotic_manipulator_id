import sympy
import sympybotics
from process_code_str import c_to_stan, create_model_block
# rbtdef = sympybotics.RobotDef('Example Robot', # robot name
#                               [('-pi/2', 0, 0, 'q+pi/2'),  # list of tuples with Denavit-Hartenberg parameters
#                                ( 'pi/2', 0, 0, 'q-pi/2')], # (alpha, a, d, theta)
#                               dh_convention='standard' # either 'standard' or 'modified'
#                              )

# a really simple one
pi = sympy.pi
q = sympybotics.robotdef.q
# (alpha, a, d, theta) = sympybotics.robotdef.default_dh_symbols
# a1, l1_x, d3, d4 = sympy.symbols('a1, l1_x, d3, d4')
# a = sympybotics.robotdef._joint_symb

# a1 = sympy.symbols('a1')
a1 = 0.8

l1 = 0.8        # length of first arm
l2 = 0.4

rbtdef = sympybotics.robotdef.RobotDef('Example Robot', # robot name
                            [('pi/2', 0, 0, 'q'),  # list of tuples with Denavit-Hartenberg parameters
                               (0, a1, 0, 'q')], # (alpha, a, d, theta)
                              dh_convention='modified' # either 'standard' or 'modified'
                             )


rbtdef.frictionmodel = {'viscous'} # options are None or a combination of 'Coulomb', 'viscous' and 'offset'
rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, -9.81]) # optional, this is the default value

rbtdef.dynparms()       # parameters

rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)

rbt.geo.T[-1]
rbt.kin.J[-1]

# C function generation
tau_C = sympybotics.robotcodegen.robot_code_to_func('C', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)

print(c_to_stan(tau_C, 2, len(rbtdef.dynparms())))

# Python code generation

tau_python = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)

print(tau_python)



rbt.calc_base_parms()
rbt.dyn.baseparms

# how to get forward dynamics
# from sympybotics.symcode import Subexprs, code_to_func
# from sympybotics.dynamics.rne import rne_forward
#
# dof = rbt.dof
# for i in range(dof):
#     g_se = Subexprs()
#     fw_code = g_se.get(rbt.geo.T[i])
#     T_str = sympybotics.robotcodegen.robot_code_to_func('python', fw_code, 'T_out', 'T_'+str(i), rbtdef)
#


print(rbt._codes)

c_str = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.c_code, 'c_out', 'corriolis_term', rbtdef)
print(c_str)

C_str = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.C_code, 'C_out', 'corriolis_matrix', rbtdef)
print(C_str)

M_str = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.M_code, 'M_out', 'mass_matrix', rbtdef)
print(M_str)

g_str = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.g_code, 'g_out', 'gravity_term', rbtdef)
print(g_str)

rbtdef.L_funcof_I


print(create_model_block(2, len(rbtdef.dynparms()),tau_C))