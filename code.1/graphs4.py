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
    a1 = 1    #Symbol('a_1')
    a2 = 0     #Symbol('a_2')
    h = 1    # Symbol('h')
    v = 1    # Symbol('v')
    d = 1    # Symbol('d')

    n = 4*np.pi

    # kx = np.linspace(-n, n, 200)
    # ky = np.linspace(-n, n, 200)

    kx = np.arange(-n, n, np.pi/20)
    ky = np.arange(-n, n, np.pi/20)

    E1 = np.sqrt((h**2)*(v**2)*(kx**2+ky**2))
    # E0y = h*v*np.absolute(ky)
    E0y = h*v*ky

    E2 = calc_E(a1, a2, E1, h, v, d, kx, ky)

    data = pd.DataFrame({'ky': ky, 'E2': E2})
    # data = data[data['E2']<4]
    # data = data[(data['E2']<-10) & (data['E2']>10)]
    
    print (data.head())

    ky_pi = ky%np.pi
    data['pi'] = ky_pi

    # data = data[data['pi']>0.3]

    # df = data[data['pi']==0]
    # print (df.head())
    # raise SystemExit
    # in_array = np.linspace(-n, np.pi, n) 
    # out_array = 1/np.tan(in_array) 

    plt.figure(figsize=(6, 12))
    plt.xlabel('ky', size=14)
    plt.ylabel('E', size=14)

    splits = split_data(data)
    print (len(splits))
    cmap = get_cmap(len(splits))

    for i, df in enumerate(splits):
        # print (df.head())
        plt.plot(df['ky'].tolist(), df['E2'].tolist(), color=cmap(i+1), marker='o')
    # plt.plot(data['ky'].tolist(), data['E2'].tolist(), color='b', marker='o')
    # plt.plot(splits[0]['ky'].tolist(), splits[0]['E2'].tolist(), 'b-')
    # plt.plot(splits[1]['ky'].tolist(), splits[1]['E2'].tolist(), 'g-')
    # plt.plot(splits[2]['ky'].tolist(), splits[2]['E2'].tolist(), 'r-')

    plt.plot(ky, E0y, color='g', linestyle='--')
    plt.plot(ky, -E0y, color='g', linestyle='--')
    # plt.plot(ky, E1, color='r', linestyle='--')
    # plt.plot(ky, -E1, color='r', linestyle='--')
    # plt.plot(df['ky'], df['E2'], color='black', marker='o')
    plt.show()