from sympy.plotting import (plot, plot_parametric,plot_implicit,
                            plot3d,plot3d_parametric_line,
                            plot3d_parametric_surface)
from sympy import *
from sympy.solvers import solve

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math
import pprint
from scipy.interpolate import interp1d, splrep, splev

from shapely import *


def calc_E1(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tan(kx*d))
    return f


def calc_E2(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tanh(kx*d))
    return f


if __name__ == '__main__':
    a1 = .4#  Symbol('a_1')
    a2 = .6#  Symbol('a_2')
    h = 1   #Symbol('h')
    v = 1   #Symbol('v')
    d = 2   #Symbol('d')

    # n = 5*np.pi

    n = 6

    kx = np.linspace(-n, n, 1000)
    ky = np.linspace(-n, n, 1000)



    E0 = v*h*ky


    result = []


    for k_y in ky:
        for k_x in [1]:
            f2 = np.sqrt((h**2)*(v**2)*(k_x**2+k_y**2))
            f1 = calc_E1(a1, a2, f2, h, v, d, k_x, k_y)

            f3 = np.sqrt((h**2)*(v**2)*(-k_x**2+k_y**2))
            f4 = calc_E2(a1, a2, f3, h, v, d, k_x, k_y)
            result.append({'kx':k_x, 'ky':k_y, 'E1':f1, 'E2':f4})

    df = pd.DataFrame(result)
    
    print (len(df))

    df.to_csv('e.csv', sep='\t', index=None, header=None)

    # df['n'] = df.apply(lambda x: np.abs(np.abs(x['E1'])-np.abs(x['E2'])), axis=1)
    # df = df[df['n']<.2]


    # df.to_csv('e2.csv', sep='\t', index=None, header=None)

    df['point'] = list(zip(df['k_y'], df['E1']))
    print (df.head())
