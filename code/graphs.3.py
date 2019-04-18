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


def E1(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/np.tan(k_x*d))+(b_p-b_m)*k_y)
    return f


def E2(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/np.tanh(k_x*d))+(b_p-b_m)*k_y)
    return f


def EE1(h, V, k_x, k_y):
    f = V*h*np.sqrt(k_x**2 + k_y**2)
    return f


def EE2(h, V, k_x, k_y):
    f = V*h*np.sqrt(k_x**2 + k_y**2)
    return f


def points_to_dataset(E, k_x, k_y):
    data = pd.DataFrame({'E':E, 'k_x':k_x, 'k_y':k_y})
    return data


def dist(k_x, k_y, E1, E2):
    E1 = points_to_dataset(E1, k_x, k_y)
    E2 = points_to_dataset(E2, k_x, k_y)
    E = E1.merge(E2, on=['k_x', 'k_y'])
    E['dist'] = E['E_x']-E['E_y']
    E['dist'] = E['dist'].abs()
    # E = E[['dist', 'k_x', 'k_y']]
    E = E.groupby(['k_y'])['dist'].min().reset_index()
    E = E.sort_values(by=['dist'])
    E = E[E['dist']<.2]
    # E = E.head(60)
    E2 = E2.merge(E, on=['k_y'])
    # E2 = E2[(E2['E']>.2) & (E2['E']<-.2)]
    return E2[['E', 'k_y']]


def interpol(x, y, k_y):
    tck = splrep(x, y, s=1)
    ynew = splev(k_y, tck, der=0)
    return ynew


def main(k_x, k_y, n, type='2D', save_name='points.xlsx'):

    writer = pd.ExcelWriter(save_name, engine='xlsxwriter')

    h = 1
    V = 1
    d = 1
    b_p = .4
    b_m = .6

    v = np.linspace(-n, n, 100)

    #################################################################################
    E1_data = E1(h, V, d, b_p, b_m, k_x, k_y)
    EE1_1_data = EE1(h, V, k_x, k_y)
    EE1_2_data = -EE1(h, V, k_x, k_y)

    E1_concat = dist(k_x, k_y, E1_data, EE1_1_data)
    E1_concat = E1_concat[E1_concat['E']>1]
    E1_concat.to_excel(writer, index=False, sheet_name='E1_concat')
    E1_concat = interpol(E1_concat['k_y'].values, E1_concat['E'].values, k_y)

    E1_concat2 = dist(k_x, k_y, E1_data, EE1_2_data)
    # E1_concat2 = E1_concat2[E1_concat['E']<-1]
    E1_concat2.to_excel(writer, index=False, sheet_name='E1_concat2')
    E1_concat2 = interpol(E1_concat2['k_y'].values, E1_concat2['E'].values, k_y)
    #################################################################################
    E2_data = E2(h, V, d, b_p, b_m, k_x, k_y)
    EE2_1_data = EE2(h, V, k_x.imag, k_y)
    EE2_2_data = -EE2(h, V, k_x.imag, k_y)
    print (EE2_1_data)
    print (EE2_2_data)

    E2_concat = dist(k_x, k_y, E1_data, EE2_1_data)
    E2_concat = E2_concat[E2_concat['E']>1]
    E2_concat.to_excel(writer, index=False, sheet_name='E2_concat')
    E2_concat = interpol(E2_concat['k_y'].values, E2_concat['E'].values, k_y)

    E2_concat2 = dist(k_x, k_y, E1_data, EE2_2_data)
    E2_concat2 = E2_concat2[E2_concat2['E']<-2]
    E2_concat2.to_excel(writer, index=False, sheet_name='E2_concat2')
    E2_concat2 = interpol(E2_concat2['k_y'].values, E2_concat2['E'].values, k_y)

    E2_concat3 = dist(k_x, k_y, E1_data, E2_data)
    E2_concat3.to_excel(writer, index=False, sheet_name='E2_concat3')
    E2_concat3 = interpol(E2_concat3['k_y'].values, E2_concat3['E'].values, k_y)

    writer.save()
    #################################################################################

    if type=='2D':

        fig = plt.figure()
        ax =fig.add_subplot(1,1,1)
        ax.grid()

        # plt.plot(k_y, E1_data)
        # plt.plot(k_y, EE1_1_data)
        # plt.plot(k_y, EE1_2_data)
        plt.plot(k_y, E1_concat, color='red')
        plt.plot(k_y, E1_concat2, color='red')

        # plt.plot(k_y, E2_data)
        # plt.plot(k_y, EE2_1_data)
        # plt.plot(k_y, EE2_2_data)
        plt.plot(k_y, E2_concat, color='red')
        plt.plot(k_y, E2_concat2, color='red')
        plt.plot(k_y, E2_concat3,'r--' )

        plt.xlabel('k_y')
        plt.ylabel('E')

        plt.ylim(-4, 4)

        # plt.xlim(0, n)
        plt.savefig('plot.png')

        # data = points_to_dataset(E, k_x, k_y)
        # print (data[(data['k_y']>-.001) & (data['k_y']<.001)])

        plt.show()


if __name__ == "__main__":
    # k_x = 1
    # k_y = 1

    # k_x = Symbol('k_X')
    # k_y = Symbol('k_y')

    n = 4

    k_x = np.linspace(-n, n, 1500)
    k_y = np.linspace(-n, n, 1500)

    main(k_x, k_y, n, type='2D')

