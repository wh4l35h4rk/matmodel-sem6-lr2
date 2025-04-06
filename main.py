import matplotlib.pyplot as plt
import numpy as np


def func_v(t, v, k, omega, phi, omega_f, A):
    return -k * v - omega**2 * phi + A * np.sin(omega_f * t)

def func_phi(t, phi, v):
    return v


def rungeKutta(x0, h, y0_1, param_list_1,
                      y0_2, param_list_2):
    print(param_list_1, param_list_2)

    x_list = [x0]
    y1 = y0_1
    y2 = y0_2
    y1_list = [y0_1]
    y2_list = [y0_2]

    while 1:
        y1_prev = y1
        y2_prev = y2

        k1 = h * func_v(x0, y1, *param_list_1)
        k2 = h * func_v(x0 + 0.5 * h, y1 + 0.5 * k1, *param_list_1)
        k3 = h * func_v(x0 + 0.5 * h, y1 + 0.5 * k2, *param_list_1)
        k4 = h * func_v(x0 + h, y1 + k3, *param_list_1)

        m1 = h * func_phi(x0, y2, *param_list_2)
        m2 = h * func_phi(x0 + 0.5 * h, y2 + 0.5 * m1, *param_list_2)
        m3 = h * func_phi(x0 + 0.5 * h, y2 + 0.5 * m2, *param_list_2)
        m4 = h * func_phi(x0 + h, y1 + m3, *param_list_2)

        y1 = y1 + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y2 = y2 + (1/6) * (m1 + 2 * m2 + 2 * m3 + m4)
        x0 = x0 + h

        x_list.append(x0)
        y1_list.append(y1)
        y2_list.append(y2)

        print(abs(y1 - y1_prev), " / ", abs(y2 - y2_prev))
        if abs(y1 - y1_prev) < 1e-5 and abs(y2 - y2_prev) < 1e-5:
            break
    return x_list, y1_list, y2_list



if __name__ == '__main__':
    v0 = 0
    phi0 = 0.1

    k = 0
    A = 0
    omega_f = 0

    L = 1
    g = 9.8
    omega = (g / L)**0.5

    h = 0.1
    t0 = 0
    v = v0
    phi = phi0


    rk = rungeKutta(t0, h, v, [k, omega, phi, omega_f, A],
                           phi, [v])

    print(rk)

