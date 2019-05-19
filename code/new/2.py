from sympy import solve, symbols, tan, tanh, sqrt
import pandas as pd
import numpy as np
import math
from shapely.geometry import *


def calc_E1(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tan(kx*d))
    return f


def calc_E2(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tanh(kx*d))
    return f



def positive_E1(kx, ky, a1, a2, h=1, v=1, d=1, _type='sym'):
    if _type == 'num':
        return (h*v)*(kx*(a1+a2)*(1/np.tan(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'sym':
        return (h*v)*(kx*(a1+a2)*(1/tan(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'math':
        return (h*v)*(kx*(a1+a2)*(1/math.tan(kx*d))+ky*(a1-a2))/(1-a1*a2)

def positive_E2(kx, ky, h=1, v=1, _type='num'):
    if _type == 'num':
        return np.sqrt((h**2)*(v**2)*(kx**2+ky**2))
    elif _type == 'math':
        return math.sqrt((h**2)*(v**2)*(kx**2+ky**2))


def negative_E1(kx, ky, a1, a2, h=1, v=1, d=1, _type='sym'):
    if _type == 'num':
        return (h*v)*(kx*(a1+a2)*(1/np.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'sym':
        return (h*v)*(kx*(a1+a2)*(1/tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'math':
        return (h*v)*(kx*(a1+a2)*(1/math.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)


def negative_E2(kx, ky, h=1, v=1, _type='num'):
    if _type == 'num':
        return np.sqrt((h**2)*(v**2)*(ky**2-kx**2))
    elif _type == 'math':
        return math.sqrt((h**2)*(v**2)*(ky**2+kx**2))
    elif _type == 'sym':
        return sqrt((h**2)*(v**2)*(ky**2-kx**2))


def negative_E2_all(kx, ky, a1, a2, h=1, v=1, d=1, _type='math'):
    if _type == 'math':
        return math.sqrt((h**2)*(v**2)*(ky**2+kx**2)) - (h*v)*(kx*(a1+a2)*(1/math.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'num':
        return np.sqrt((h**2)*(v**2)*(ky**2+kx**2)) - (h*v)*(kx*(a1+a2)*(1/np.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)


def _solve(e1, e2, pararmetr):
    return solve([e1, e2], pararmetr)


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
    a1 = 1
    a2 = 0
    kx = np.arange(-6, 6, .1)
    ky = np.arange(-6, 6, .1)


    # result = []
    # for _kx in kx:
    #     for _ky in ky:
    #         e2 = positive_E2(_kx, _ky, _type='math')
    #         e1 = positive_E1(_kx, _ky, a1, a2, _type='math')
    #         result.append({'e1': e1, 'e2': e2, 'ky': _ky})

    # result = pd.DataFrame(result)

    # result = intersections(result)
    # result.to_csv("result/math_p1_intersect", sep='\t', index=None)

    result = []
    for _ky in ky:
        for _kx in kx:
            # print (_kx, 1j*_kx, (1j*_kx)**2)
            try:
                # e2 = negative_E2(1j*_kx, _ky, _type='math')
                e1 = negative_E1(1j*_kx, _ky, a1, a2, _type='math')
                print (e1)
            except ValueError:
                pass
            # print (e1)
            # result.append({'e1': e1, 'e2': e2, 'ky': _ky})

    # result = pd.DataFrame(result)

    # result = intersections(result)
    # result.to_csv("result/math_p2_intersect", sep='\t', index=None) 