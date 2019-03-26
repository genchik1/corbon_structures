from sympy.plotting import (plot, plot_parametric,plot_implicit,
                            plot3d,plot3d_parametric_line,
                            plot3d_parametric_surface)
from sympy import *
from sympy.solvers import solve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pprint

from mpl_toolkits import mplot3d


def func_1(k_x, k_y, h, V, d, b_p, b_m, parameter=1):
    if parameter == 1:
        return (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(np.cos(k_x*d)/np.sin(k_x*d))+(b_p-b_m)*k_y)
    else:
        return (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/np.tanh(k_x*d))+(b_p-b_m)*k_y)

 
def func_2(k_x, k_y, h, V, d, b_p, b_m, parameter=1):
    if parameter == 1:
        KX = k_x*k_x
    else:
        KX = (-1) * k_x*k_x
    return (-1)*V*h*np.sqrt(KX + k_y*k_y)


def func_3(k_x, k_y, h, V, d, b_p, b_m, parameter=1):
    if parameter == 1:
        KX = k_x*k_x
    else:
        KX = (-1) * k_x*k_x
    return V*h*np.sqrt(KX + k_y*k_y)


def calculate(f, k_x, k_y, h, V, d, b_p, b_m, parameter=1):
    E_array = np.array([])
    KX_array = np.array([])
    KY_array = np.array([])

    print(type(k_x))
    for ky in k_y:
        for kx in k_x:
            x = f(kx, ky, h, V, d, b_p, b_m, parameter)
            E_array = np.append(E_array, [x])
            KY_array = np.append(KY_array, [ky])
            KX_array = np.append(KX_array, [kx])
    return E_array, KX_array, KY_array


if __name__ == "__main__":
    h = 1
    V = 1
    d = 2
    b_p = .5
    b_m = .1
  
    n = 10

    v = np.linspace(-n, n, 100)

    k_x = 1

    k_x = Symbol('k_X')
    k_y = Symbol('k_y')

    # k_x = np.linspace(-n, n, 100)
    # k_y = np.linspace(-n, n, 100)


    E1 = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/tan(k_x*d))+(b_p-b_m)*k_y)
    E2 = (-1)*V*h*sqrt(k_x*k_x + k_y*k_y)
    E3 = V*h*sqrt(k_x*k_x + k_y*k_y)

    E4 = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/tanh(k_x*d))+(b_p-b_m)*k_y)

    E5 = (-1)*V*h*sqrt((-1)*(k_x*k_x) + k_y*k_y) 
    E6 = V*h*sqrt((-1)*(k_x*k_x) + k_y*k_y)

    # plot( E5, E6,(k_y,-n,n,0.01))
    plot3d(E4, (k_y,-n,n,0.0001), (k_x,-n,n,0.0001))