import sympy
import numpy
from math import cos, sin

import sympybotics
from sympybotics._compatibility_ import exec_





def test_scara_dh_sym_geo_kin():

    pi = sympy.pi
    q = sympybotics.robotdef.q

    a1, a2, d3, d4 = sympy.symbols('a1, a2, d3, d4')

    scara = sympybotics.robotdef.RobotDef(
        'SCARA - Spong',
        [( 0, a1,  0, q),
         (pi, a2,  0, q),
         ( 0,  0,  q, 0),
         ( 0,  0, d4, q)],
        dh_convention='standard')

    scara_geo = sympybotics.geometry.Geometry(scara)
    scara_kin = sympybotics.kinematics.Kinematics(scara, scara_geo)

    cos, sin = sympy.cos, sympy.sin
    q1, q2, q3, q4 = sympy.flatten(scara.q)

    T_spong = sympy.Matrix([
        [(-sin(q1)*sin(q2) + cos(q1)*cos(q2))*cos(q4) + (sin(q1)*cos(q2) +
                                                         sin(q2)*cos(q1))*sin(q4), -(-sin(q1)*sin(q2) +
                                                                                     cos(q1)*cos(q2))*sin(q4) + (sin(q1)*cos(q2) +
                                                                                                                 sin(q2)*cos(q1))*cos(q4), 0, a1*cos(q1) - a2*sin(q1)*sin(q2) +
         a2*cos(q1)*cos(q2)],
        [(sin(q1)*sin(q2) - cos(q1)*cos(q2))*sin(q4) + (sin(q1)*cos(q2) +
                                                        sin(q2)*cos(q1))*cos(q4), (sin(q1)*sin(q2) -
                                                                                   cos(q1)*cos(q2))*cos(q4) - (sin(q1)*cos(q2) +
                                                                                                               sin(q2)*cos(q1))*sin(q4), 0, a1*sin(q1) + a2*sin(q1)*cos(q2) +
         a2*sin(q2)*cos(q1)],
        [0, 0, -1, -d4 - q3],
        [0, 0, 0, 1]])

    J_spong = sympy.Matrix([[-a1*sin(q1) - a2*sin(q1)*cos(q2) -
                             a2*sin(q2)*cos(q1), -a2*sin(q1)*cos(q2) -
                             a2*sin(q2)*cos(q1), 0, 0],
                            [a1*cos(q1) - a2*sin(q1)*sin(q2) +
                             a2*cos(q1)*cos(q2), -a2*sin(q1)*sin(q2) +
                             a2*cos(q1)*cos(q2), 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 1, 0, -1]])

    assert (scara_geo.T[-1] - T_spong).expand() == sympy.zeros(4)
    assert (scara_kin.J[-1] - J_spong).expand() == sympy.zeros(6, 4)

    scara.gravityacc = sympy.Matrix([0.0, 0.0, -9.81]) # optional, this is the default value
    print('1')
    # rbtdef.gravityacc = sympy.Matrix([-0.81, 0, 0.0]) # optional, this is the default value

    scara.dynparms()       # parameters
    print('2')
    rbt = sympybotics.RobotDynCode(scara, verbose=True)
    print('test sucessful')

test_scara_dh_sym_geo_kin()