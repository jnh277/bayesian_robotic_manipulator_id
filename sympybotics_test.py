import sympy
import sympybotics
rbtdef = sympybotics.RobotDef('Example Robot', # robot name
                              [('-pi/2', 0, 0, 'q+pi/2'),  # list of tuples with Denavit-Hartenberg parameters
                               ( 'pi/2', 0, 0, 'q-pi/2')], # (alpha, a, d, theta)
                              dh_convention='standard' # either 'standard' or 'modified'
                             )
rbtdef.frictionmodel = {'Coulomb', 'viscous'} # options are None or a combination of 'Coulomb', 'viscous' and 'offset'
rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, -9.81]) # optional, this is the default value

rbtdef.dynparms()       # parameters

rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)

rbt.kin.J[-1]

# C function generation
tau_C = sympybotics.robotcodegen.robot_code_to_func('C', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)

print(tau_C)

# Python code generation

tau_python = sympybotics.robotcodegen.robot_code_to_func('Python', rbt.invdyn_code, 'tau_out', 'tau', rbtdef)

print(tau_python)

rbt.calc_base_parms()
rbt.dyn.baseparms