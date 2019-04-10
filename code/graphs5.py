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


def calc_E(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tan(kx*d))
    return f


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def split_data(data):
    x_index = data.index[data['pi']<=.1].tolist()
    print (x_index)
    split_datas = []
    x_i = 0
    for x in x_index:
        df = data.iloc[x_i:x-1]
        df = df[(df['E2']>-10)&(df['E2']<10)]
        df = df[(df['pi']>.1)]
        print (df.head())
        if len(df)>6:
            split_datas.append(df)
        x_i = x+1
    return split_datas
    

if __name__ == '__main__':
    a1 = 1#  Symbol('a_1')
    a2 = 0#  Symbol('a_2')
    h = 1   #Symbol('h')
    v = 1   #Symbol('v')
    d = 1   #Symbol('d')

    n = 4*np.pi

    kx = 1 #np.linspace(-n, n, 200)
    ky = np.linspace(-n, n, 200)

    # kx = np.arange(-n, n, np.pi/20)
    # ky = np.arange(-n, n, np.pi/20)

    # kx = 1#Symbol('k_x')
    # ky = Symbol('k_y')


    E1 = np.sqrt((h**2)*(v**2)*(kx**2+ky**2))

    E = v*h*ky
    
    N = pi/2

    data = pd.DataFrame({'ky':ky, 'E1':calc_E(a1, a2, E1, h, v, d, kx, ky), 'E0':E})

    print (data.head())

    data.to_csv('e.csv', sep='\t', index=None, header=None)

    raise SystemExit

    # p1=plot(calc_E(a1, a2, E1, h, v, d, kx, ky),(ky,-N,N), line_color='b', show=False) 
    # p2=plot(-calc_E(a1, a2, E1, h, v, d, kx, ky),(ky,-N,N), line_color='r', show=False) 
    # p3=plot(E,(ky,-N,N), line_color='g', show=False) 
    # p4=plot(-E,(ky,-N,N), line_color='g', show=False)


    # p1.extend(p1)
    # p1.extend(p2)
    # p1.extend(p3)
    # p1.extend(p4)

    # p1.show()