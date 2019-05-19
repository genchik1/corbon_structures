import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
# from shapely.geometry import *


def f(k_y, k_x, a):
    return 2*np.cos(np.sqrt(3)*k_y*a)+4*np.cos(np.sqrt(3)*k_y*a/2)*np.cos(3*k_x*a/2)


def E(t1, t2, k_y, k_x, a):
    return t1*np.sqrt(3+f(k_y, k_x, a))-t2*f(k_y, k_x, a)


def E2(t1, t2, k_y, k_x, a):
    return -t1*np.sqrt(3+f(k_y, k_x, a))-t2*f(k_y, k_x, a)

if __name__ == '__main__':

    kx = np.arange(-5, 5, 0.001)
    ky = np.arange(-5, 5, 0.001)

    t1 = 2.7
    t2 = -.2*t1

    a = 1.42


    # result = []


    # for k_x in kx:
    #     for k_y in ky:
    #         e1 = E(t1, t2, k_y, k_x, a)
    #         e2 = E2(t1, t2, k_y, k_x, a)
    #         # if not np.isnan(e1) and  not np.isnan(e2):
    #         result.append({'kx':k_x, 'ky':k_y, 'e1':e1, 'e2':e2})


    # electronic_dispersion = pd.DataFrame(result)

    # ky, kx = np.meshgrid(ky, kx)

    Z = E(t1, t2, ky, kx, a)
    Z2 = E2(t1, t2, ky, kx, a)

    electronic_dispersion = pd.DataFrame({'kx':kx, 'ky':ky, 'e1':Z, 'e2': Z2})

    # print (electronic_dispersion.head())

    electronic_dispersion.to_csv('result/electronic_dispersion', sep='\t', index=None)


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')


    # surf = ax.plot_surface(kx, ky, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.plot_wireframe(kx, ky, Z)
    # plt.plot(kx, ky)
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
