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


def calc_E1(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tan(kx*d))
    return kx, ky, f


def calc_E2(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tanh(kx*d))
    return f


if __name__ == '__main__':
    a1 = 1#  Symbol('a_1')
    a2 = 0#  Symbol('a_2')
    h = 1   #Symbol('h')
    v = 1   #Symbol('v')
    d = 1   #Symbol('d')

    # n = 5*np.pi

    n = 6

    kx = np.linspace(-n, n, 1000)
    ky = np.linspace(-n, n, 1000)


    E1 = np.sqrt((h**2)*(v**2)*(kx**2+ky**2))
    E2 = np.sqrt((h**2)*(v**2)*(-(kx**2)+ky**2))

    E0 = v*h*ky

    _kx, _ky, _E = calc_E1(a1, a2, E1, h, v, d, kx, ky)

    df = pd.DataFrame({
        '_kx':_kx,
        '_ky':_ky,
        '_E':_E,
    })
    
    # data = pd.DataFrame({'ky':ky,
    #                      'E1':,
    #                      'E2':calc_E2(a1, a2, E2, h, v, d, -kx, ky),
    #                      'E01':E0,
    #                      'E02':-E0})



    print (df.head())
    print (len(df))

    # data.to_csv('e.csv', sep='\t', index=None, header=None)
