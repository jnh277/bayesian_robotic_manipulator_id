import sympy
import sympybotics
from process_code_str import create_model_block

from process_code_str import c_to_stan
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

d0 = sympy.symbols('d0')
a1 = sympy.symbols('a1')
# d0 = 0.0
# a1 = 0.8

l1 = 0.8        # length of first arm
l2 = 0.4

rbtdef = sympybotics.robotdef.RobotDef('Example Robot', # robot name
                            [(0, 0, d0, 'q'),        # rotating base
                             ('pi/2', 0, 0, 'q'),  # list of tuples with Denavit-Hartenberg parameters
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

print(tau_C)

# Python code generation

tau_python = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)

print(tau_python)



# rbt.calc_base_parms()
# rbt.dyn.baseparms


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


print(create_model_block(3, len(rbtdef.dynparms()),tau_C))