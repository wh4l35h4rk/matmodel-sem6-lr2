import matplotlib.pyplot as plt
import numpy as np

a = 0
b = 10
N = 400
h = (b - a) / N


def dv(t, v, phi, omega, k, omega_f, A):
    return -k * v - omega**2 * np.sin(phi) + A * np.sin(omega_f * t)

def dphi(v):
    return v


def func(t, func_values, omega, k=0, omega_f=0, A=0):
    v = func_values[0]
    phi = func_values[1]
    return np.array([dv(t, v, phi, omega, k, omega_f, A),
                     dphi(v)])


def rungeKutta(x, start_conditions, func, param_list):
    y = np.zeros((1, len(start_conditions)))
    y[0] = start_conditions

    for i in range(len(x) - 1):
        k1 = func(x[i], y[i], *param_list)
        k2 = func(x[i] + 0.5*h, y[i] + 0.5*h * k1, *param_list)
        k3 = func(x[i] + 0.5*h, y[i] + 0.5*h * k2, *param_list)
        k4 = func(x[i] + h, y[i] + h * k3, *param_list)

        y_new = y[i] + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.append(y, [y_new], axis=0)
    return y



if __name__ == '__main__':
    v0 = 0
    phi0 = 0.1

    k = 0
    A = 0
    omega_f = 0

    L = 1
    g = 9.8
    omega = (g / L)**0.5

    t = np.linspace(a, b, N + 1)

    parameters = [omega, k, omega_f, A]
    y = rungeKutta(t, [v0, phi0], func, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, y[:, 1])
    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"$\phi$, рад")
    axs[0].grid()
    axs[1].plot(y[:, 1], y[:, 0])
    axs[1].set_xlabel(r"$\phi$, рад")
    axs[1].set_ylabel(r"$v$, рад/с")
    axs[1].grid()
    plt.show()

