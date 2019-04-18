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
    b_p = .9
    b_m = .1
  
    n = 4

    v = np.linspace(-n, n, 100)

    k_x = np.linspace(-n, n, 100)
    k_y = np.linspace(-n, n, 100)

    E_1, KX_array, KY_array = calculate(func_1, k_x, k_y, h, V, d, b_p, b_m)
    E_2, _, _ = calculate(func_2, k_x, k_y, h, V, d, b_p, b_m)
    E_3, _, _ = calculate(func_3, k_x, k_y, h, V, d, b_p, b_m)

    
    E_1_, _, _ = calculate(func_1, v, k_y, h, V, d, b_p, b_m, 2)
    E_2_, _, _ = calculate(func_2, v, k_y, h, V, d, b_p, b_m, 2)
    E_3_, _, _ = calculate(func_3, v, k_y, h, V, d, b_p, b_m, 2)

    # df = pd.DataFrame({'x':k_y, 'e1':E_1, 'e3':E_3})
    # df = df[df['e1']==df['e3']]
    # print (df)


    # 2D:
    # fig, ax = plt.subplots()
    # plt.plot(KY_array, E_1)
    # plt.plot(KY_array, E_2)
    # plt.plot(KY_array, E_3)
    # # plt.plot(df['x'], df['e1'], '--r')


    # plt.plot(KY_array, E_1_)
    # plt.plot(KY_array, E_2_)
    # plt.plot(KY_array, E_3_)

    # ax.grid()
    # plt.show()


    # 3D:
    
    

    

    fig = plt.figure()
    ax =fig.add_subplot(projection='3d')
    # ax = plt.axes(projection='3d')
    # ax.contour3D(KX_array, KY_array, E_1, 50, cmap='binary')
    ax.plot3D(KX_array, KY_array, E_1)
    ax.plot3D(KX_array, KY_array, E_2)
    plt.show()