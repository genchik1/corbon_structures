from sympy import solve, symbols, tan, tanh, sqrt
import pandas as pd
import numpy as np
import math
from shapely import *


def calc_E1(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tan(kx*d))
    return f


def calc_E2(a1, a2, E, h, v, d, kx, ky):
    f = (1-a1*a2)*(E/h*v)+(a2-a1)*ky - (a1+a2)*kx*(1/np.tanh(kx*d))
    return f



def positive_E1(E, kx, ky, a1, a2, h=1, v=1, d=1, _type='sym'):
    if _type == 'num':
        return E - (h*v)*(kx*(a1+a2)*(1/np.tan(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'sym':
        return E - (h*v)*(kx*(a1+a2)*(1/tan(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'math':
        return E - (h*v)*(kx*(a1+a2)*(1/math.tan(kx*d))+ky*(a1-a2))/(1-a1*a2)

def positive_E2(kx, ky, h=1, v=1, _type='num'):
    if _type == 'num':
        return np.sqrt((h**2)*(v**2)*(kx**2+ky**2))
    elif _type == 'math':
        return math.sqrt((h**2)*(v**2)*(kx**2+ky**2))


def negative_E1(E, kx, ky, a1, a2, h=1, v=1, d=1, _type='sym'):
    if _type == 'num':
        return E - (h*v)*(kx*(a1+a2)*(1/np.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'sym':
        return E - (h*v)*(kx*(a1+a2)*(1/tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'math':
        return E - (h*v)*(kx*(a1+a2)*(1/math.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)


def negative_E2(kx, ky, h=1, v=1, _type='num'):
    if _type == 'num':
        return np.sqrt((h**2)*(v**2)*(ky**2-kx**2))
    elif _type == 'math':
        return math.sqrt((h**2)*(v**2)*(ky**2+kx**2))
    elif _type == 'sym':
        return sqrt((h**2)*(v**2)*(ky**2-kx**2))


def negative_E2_all(kx, ky, a1, a2, h=1, v=1, d=1, _type='math'):
    print(kx)
    if _type == 'math':
        return math.sqrt((h**2)*(v**2)*(ky**2+kx**2)) - (h*v)*(kx*(a1+a2)*(1/math.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)
    elif _type == 'num':
        return np.sqrt((h**2)*(v**2)*(ky**2+kx**2)) - (h*v)*(kx*(a1+a2)*(1/np.tanh(kx*d))+ky*(a1-a2))/(1-a1*a2)


def _solve(e1, e2, pararmetr):
    return solve([e1, e2], pararmetr)


if __name__ == '__main__':
    a1 = 1
    a2 = 0
    kx = np.arange(-6, 6, .06)
    ky = np.arange(-6, 6, .06)


    # e2 = positive_E2(kx, ky)
    # e1 = positive_E1(e2, kx, ky, a1, a2, _type='num')

    # result = pd.DataFrame({'E':e1, 'ky': ky})
    # result.to_csv("result/num_p1", sep='\t', index=None)


    # result = []
    # for _kx in kx:
    #     for _ky in ky:
    #         e2 = positive_E2(_kx, _ky, _type='math')
    #         e1 = positive_E1(e2, _kx, _ky, a1, a2, _type='math')
    #         result.append({'E1': e1, 'ky': _ky})

    # result = pd.DataFrame(result)
    # result.to_csv("result/math_p1", sep='\t', index=None)

    result = []
    for _kx in kx:
        for _ky in ky:
            try:
                e2 = math.sqrt(_ky**2-_kx**2)
                print ('e2', e2)
                e1 = e2-(1/math.tanh(1))-_ky
                print ('e1', e1)
                result.append({'E1': e1, 'ky': _ky})
            except TypeError:
                pass
            except ValueError:
                pass

    result = pd.DataFrame(result)
    result.to_csv("result/math_n1", sep='\t', index=None)

    #### e2 = negative_E2(kx, ky)
    #### print (e2)
    #### e1 = negative_E1(e2, kx.imag, ky, a1, a2, _type='num')

    #### result = pd.DataFrame({'E':e1, 'ky': ky})
    #### result.to_csv("result/num_p2", sep='\t', index=None)

    #### result = []
    #### for _kx in kx:
    ####     for _ky in ky:
    ####         try:
    ####             print (_kx, _ky)
    ####             e = negative_E2_all(1j*_kx, _ky, a1, a2, _type='math')
    ####             print (e)
    ####             # e1 = negative_E1(e2, 1j*_kx, _ky, a1, a2, _type='math')
    ####             # result.append({'E1': e1, 'ky': _ky})
    ####         except ZeroDivisionError:
    ####             pass

    #### result = pd.DataFrame(result)
    #### result.to_csv("result/math_p2", sep='\t', index=None)


    # negative_E2_all(kx.item, ky, a1, a2, _type='math')


