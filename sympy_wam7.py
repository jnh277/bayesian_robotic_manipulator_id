import sympy
import sympybotics
from process_code_str import c_to_stan,create_stan_model
# rbtdef = sympybotics.RobotDef('Example Robot', # robot name
#                               [('-pi/2', 0, 0, 'q+pi/2'),  # list of tuples with Denavit-Hartenberg parameters
#                                ( 'pi/2', 0, 0, 'q-pi/2')], # (alpha, a, d, theta)
#                               dh_convention='standard' # either 'standard' or 'modified'
#                              )

# WAM7 robot

rbtdef = sympybotics.RobotDef("WAM Arm 7 DOF",
            [("-pi/2", 0, 0, "q"),
             ("pi/2", 0, 0, "q"),
             ("-pi/2", 0.045, 0.55, "q"),
             ("pi/2", -0.045, 0, "q"),
             ("-pi/2", 0, 0.3, "q"),
             ("pi/2", 0, 0, "q"),
             (0, 0, 0.06, "q")],
            dh_convention="standard")

rbtdef.frictionmodel = {'Coulomb', 'viscous', 'offset'}
rbtdef.driveinertiamodel = 'simplified'


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

create_stan_model(rbt.dof, len(rbtdef.dynparms()), tau_C, "stan/wam7.stan",frictionmodel={'viscous','Coulomb','offset'},driveinertiamodel='simplified')
