import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function
from math import sin, cos, radians
import timeit

'''
A simple Python benchmark.

Results on an overclocked AMD FX-8150 Eight-Core CPU @ 3.0 GHz, and
an Intel Core i5-2410M CPU @ 2.30GHz.

$ python -OO bench.py
1.99843406677 2.00139904022 2.0145778656
2.38226699829 2.38675498962 2.38853287697

$ python3 -OO bench.py
2.2073315899979207 2.2098999509980786 2.222747125000751
2.273064840992447  2.274112678001984 2.2759074380010134

$ pypy -OO bench.py
0.245079994202 0.24707698822  0.247714996338
0.241708040237 0.242873907089 0.245008945465

$ pypy3 -OO bench.py
1.1291401386260986 1.1360960006713867 1.1375579833984375
1.2108190059661865 1.2172389030456543 1.2178328037261963

'''


def bench():
    product = 1.0
    for counter in range(1, 1000, 1):
        for dex in list(range(1, 360, 1)):
            angle = radians(dex)
            product *= sin(angle)**2 + cos(angle)**2
    return product

def sigmoid(x, alpha, beta):
    """Standard sigmoid function, translated by beta and scaled by alpha.
    With positive alpha, approaches 0 as x -> -inf and 1 as x -> inf.
    Equals 1/2 at x = beta and smoothly interpolates between these values.

    Parameters
    ----------
    x : float
        Value to fix between 0 - 1
    alpha : float
        Steepness of the function. Slope is alpha/4 at x = beta.
        Valid for all real values.
        Note: negative values will switch the limits.
    beta : float
        Translates function along the X-axis. Curve is centered at x = beta.
        Valid for all real values.

    Returns
    -------
    float
        Returns value constrained between 0 and 1.
    """

    arg = alpha * (x - beta)
    # if abs(arg) of exp is >100 return 0 or 1,
    # prevents over/underflow
    if arg > 100:
        return 1.0
    elif -arg > 100:
        return 0.0
    else:
        # Actual sigmoid function for reasonable values
        return 1.0 / (1.0 + np.exp(-arg))

def asymmetric_booster(x, alpha, beta, gamma):

    """Asymmetric booster function, translated by beta, scaled by alpha, and
    pushed along Y-axis by gamma.
    With positive alpha, approaches 0 as x -> -inf and 100 as x -> inf.
    Smoothly interpolates between these values.
    Note: Equals gamma at x = beta.

    Parameters
    ----------
    x : float OR np.array(float)
        Input stress (or array of stress values) to fix between 0 - 100
    alpha : float
        Steepness of the function. Slope is 25*alpha/2 at x = beta.
        Valid for all real values, but > 0 for typical behavior.
    beta : float
        Translates function along the X-axis. Curve is centered at x = beta
        Valid for all real values.
    gamma : float
        'Pushes' curve along Y-axis,
        pushed up for gamma>50,and down for gamma<50.
        Valid for range (25,75) excluding endpoints.

    Returns
    -------
    float
        Returns stress value, constrained between 0 and 100.
    """
    if hasattr(x, "__iter__"):
        return np.array([asymmetric_booster(_x, alpha, beta, gamma) for _x in x])

    if (gamma < 25) or (gamma > 75):
        raise Exception(f"gamma={gamma} is outside range (25,75)")

    tol = 1e-10

    if gamma < 25.0 + tol:
        gamma = 25.0 + tol
    elif 75.0 - gamma < tol:
        gamma = 75.0 - tol

    # Rescale gamma to ensure f(stress=beta) = gamma
    gamma = 2 * (gamma - 25)

    # k regulates cutover from x < beta to x > beta behavior
    # For gamma further from 50, cutover should occur more quickly,
    # pushing the curve along the Y-axis
    d = np.abs(gamma - 50) + 50
    k = 1 + np.log(d / (100 - d))

    # scale alpha such that slope at x = beta is unaffected by gamma
    alpha = alpha / ((k + 1))

    # Cutover functions
    A = sigmoid(x, alpha * k, beta)  # 1 at x >> beta, 0 at x << beta
    B = sigmoid(x, -alpha * k, beta)  # 1 at x << beta. 0 at x >> beta

    # x >> beta behavior
    f_largex = (100 - gamma) * sigmoid(x, alpha, beta) + gamma

    # x << beta behavior
    f_smallx = gamma * sigmoid(x, alpha, beta)

    score = A * f_largex + B * f_smallx

    return score
if __name__ == '__main__':
    result = timeit.repeat('bench.bench()', setup='import bench', number=10, repeat=10)
    result = list(sorted(result))
    print(*result[:3])