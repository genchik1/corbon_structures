
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import *


def calc_E1(h, V, d, b_p, b_m, kx, ky, E):
    f = (1-b_p*b_m)*(E/h*V)+(b_m-b_p)*ky - (b_p+b_m)*kx*(1/math.tan(kx*d))
    return f


def calc_E2(h, V, d, b_p, b_m, kx, ky, E):
    f = (1-b_p*b_m)*(E/h*V)+(b_m-b_p)*ky - (b_p+b_m)*kx*(1/math.tanh(kx*d))
    return f


def E1(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/np.tan(k_x*d))+(b_p-b_m)*k_y)
    return f


def E1_bub(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/math.tan(k_x*d))+(b_p-b_m)*k_y)
    return f


def E2(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/np.tanh(k_x*d))+(b_p-b_m)*k_y)
    return f


def E2_bub(h, V, d, b_p, b_m, k_x, k_y):
    f = (h*V/1-b_p*b_m)*(k_x*(b_p+b_m)*(1/math.tanh(k_x*d))+(b_p-b_m)*k_y)
    return f


def EE1(h, V, k_x, k_y):
    f = V*h*np.sqrt(k_x**2 + k_y**2)
    return f


def create_data(e1, e2, ky):
    return pd.DataFrame({'ky':ky, 'e1':e1, 'e2':e2})


def intersections(data):
    data = data.dropna()

    data.to_csv('result/test', sep='\t', index=None)

    data['Points1'] = list(zip(data['ky'], data['e1']))
    data['Points2'] = list(zip(data['ky'], data['e2']))

    Points1 = data['Points1'].apply(Point)
    Points2 = data['Points2'].apply(Point)

    l1 = LineString(Points1)
    l2 = LineString(Points2)

    result = []

    for ob in l1.intersection(l2):
        x, y = ob.xy
        result.append({'x':x[0], 'y':y[0]})

    return pd.DataFrame(result)


if __name__ == '__main__':
    n = 6
    h = 1
    V = 1
    d = 2
    b_p = .4
    b_m = .6

    kx = np.linspace(-n, n, 50)
    ky = np.linspace(-n, n, 50)


    ################################################################
    # 1

    # e1 = E1(h, V, d, b_p, b_m, kx, ky)
    # e1_1 = EE1(h, V, kx, ky)
    # df1 = create_data(e1, e1_1, ky)
    # df1 = intersections(df1)
    # df1.to_csv('result/df1', sep='\t', index=None, header=None)

    ################################################################
    #2

    # E1 = EE1(h, V, kx, ky)
    # E2 = EE1(h, V, -kx, ky)
    # e2_1 = calc_E1(b_p, b_m, E1, h, V, d, kx, ky)
    # e2_2 = calc_E2(b_p, b_m, E2, h, V, d, kx, ky)

    # df2 = create_data(e2_1, e2_2, ky)
    # df2 = intersections(df2)
    # df2.to_csv('result/df2', sep='\t', index=None, header=None)

    ################################################################

    # result = []
    # for k_y in ky:
    #     for k_x in kx:
    #         E1 = V*h*np.sqrt(k_x**2 + k_y**2)
    #         E2 = V*h*np.sqrt(-k_x**2 + k_y**2)
    #         _e1 = calc_E1(b_p, b_m, E1, h, V, d, k_x, k_y)
    #         _e2 = calc_E2(b_p, b_m, E2, h, V, d, k_x, k_y)
    #         if not np.isnan(_e1) and not np.isnan(_e2):
    #             result.append({'ky':k_y, 'e1':_e1, 'e2':_e2})
    #             print ({'ky':k_y, 'e1':_e1, 'e2':_e2})
    # df2_1 = intersections(pd.DataFrame(result))
    # df2_1.to_csv('result/df2_1', sep='\t', index=None, header=None)

    ################################################################

    # E1 = V*h*np.sqrt(kx**2 + ky**2)
    # E2 = V*h*np.sqrt(-kx**2 + ky**2)

    # _e1 = calc_E1(b_p, b_m, E1, h, V, d, kx, ky)
    # _e2 = calc_E2(b_p, b_m, E2, h, V, d, kx, ky)

    # df2_1 = intersections(pd.DataFrame({'ky':ky, 'e1': _e1, 'e2': _e2}))
    # df2_1.to_csv('result/df2_2', sep='\t', index=None, header=None)

    ################################################################

    # result = []
    # for k_y in ky:
    #     for k_x in kx:
    #         E1 = V*h*np.sqrt(k_x**2 + k_y**2)

    #         e1 = E1_bub(h, V, d, b_p, b_m, k_x, k_y)
    #         if not np.isnan(E1) and not np.isnan(e1):
    #             result.append({'ky':k_y, 'e1':e1, 'e2':E1})

    # df3_1 = intersections(pd.DataFrame(result))
    # df3_1.to_csv('result/df3_1', sep='\t', index=None, header=None)


    # result = []
    # for k_y in ky:
    #     for k_x in kx:
    #         E1 = V*h*np.sqrt(k_x**2 + k_y**2)

    #         e1 = E1_bub(h, V, d, b_p, b_m, k_x, k_y)
    #         if not np.isnan(E1) and not np.isnan(e1):
    #             result.append({'ky':k_y, 'e1':e1, 'e2':-E1})

    # df3_1 = intersections(pd.DataFrame(result))
    # df3_1.to_csv('result/df3_2', sep='\t', index=None, header=None)


    # result = []
    # for k_y in ky:
    #     for k_x in kx:
    #         E1 = V*h*np.sqrt(-k_x**2 + k_y**2)

    #         e1 = E2_bub(h, V, d, b_p, b_m, k_x, k_y)
    #         if not np.isnan(E1) and not np.isnan(e1):
    #             result.append({'ky':k_y, 'e1':e1, 'e2':E1})

    # df3_1 = intersections(pd.DataFrame(result))
    # df3_1.to_csv('result/df4_1', sep='\t', index=None, header=None)

    # result = []
    # for k_y in ky:
    #     for k_x in kx:
    #         E1 = V*h*np.sqrt(-k_x**2 + k_y**2)

    #         e1 = E2_bub(h, V, d, b_p, b_m, k_x, k_y)
    #         if not np.isnan(E1) and not np.isnan(e1):
    #             result.append({'ky':k_y, 'e1':e1, 'e2':-E1})

    # df3_1 = intersections(pd.DataFrame(result))
    # df3_1.to_csv('result/df4_2', sep='\t', index=None, header=None)



    ################################################################
    ################################################################


    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt(k_x**2 + k_y**2)
            e1 = E1_bub(h, V, d, b_p, b_m, k_x, k_y)

            if not np.isnan(E1) and not np.isnan(e1):
                result.append({'ky':k_y, 'e1':e1, 'e2':E1})

    df5_1 = intersections(pd.DataFrame(result))
    df5_1.to_csv('result/bub_df1', sep='\t', index=None, header=None)

    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt(k_x**2 + k_y**2)
            e1 = E1_bub(h, V, d, b_p, b_m, k_x, k_y)

            if not np.isnan(E1) and not np.isnan(e1):
                result.append({'ky':k_y, 'e1':e1, 'e2':-E1})

    df5_2 = intersections(pd.DataFrame(result))
    df5_2.to_csv('result/bub_df2', sep='\t', index=None, header=None)

    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt((-1)*k_x**2 + k_y**2)
            e1 = E2_bub(h, V, d, b_p, b_m, k_x, k_y)

            if not np.isnan(E1) and not np.isnan(e1):
                result.append({'ky':k_y, 'e1':e1, 'e2':E1})

    df5_2 = intersections(pd.DataFrame(result))
    df5_2.to_csv('result/bub_df3', sep='\t', index=None, header=None)

    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt((-1)*k_x**2 + k_y**2)
            e1 = E2_bub(h, V, d, b_p, b_m, k_x, k_y)

            if not np.isnan(E1) and not np.isnan(e1):
                result.append({'ky':k_y, 'e1':e1, 'e2':-E1})

    df5_2 = intersections(pd.DataFrame(result))
    df5_2.to_csv('result/bub_df4', sep='\t', index=None, header=None)


    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt(k_x**2 + k_y**2)
            E2 = -V*h*np.sqrt(k_x**2 + k_y**2)

            if not np.isnan(E1) and not np.isnan(E2):
                result.append({'ky':k_y, 'e1':E2, 'e2':E1})

    df5_2 = pd.DataFrame(result)
    df5_2.to_csv('result/bub_df5', sep='\t', index=None, header=None)


    result = []
    for k_y in ky:
        for k_x in kx:
            E1 = V*h*np.sqrt((-1)*(k_x)**2 + k_y**2)
            E2 = -V*h*np.sqrt((-1)*(k_x)**2 + k_y**2)

            if not np.isnan(E1) and not np.isnan(E2):
                result.append({'ky':k_y, 'e1':E2, 'e2':E1})

    df5_2 = pd.DataFrame(result)
    df5_2.to_csv('result/bub_df6', sep='\t', index=None, header=None)

    