from cgitb import handler
import matplotlib.pyplot as plt
import numpy as np


def dv_linear(t, v, phi, omega, k, omega_f, A):
    return -k * v - omega**2 * phi + A * np.sin(omega_f * t)

def dv_nonlinear(t, v, phi, omega, k, omega_f, A):
    return -k * v - omega**2 * np.sin(phi) + A * np.sin(omega_f * t)

def dphi(v):
    return v


def func(dv_type, t, func_values, omega, k=0, omega_f=0, A=0):
    v = func_values[0]
    phi = func_values[1]
    return np.array([dv_type(t, v, phi, omega, k, omega_f, A),
                     dphi(v)])


def rungeKutta(x, start_conditions, dv_type, param_list):
    h = (x[len(x) - 1] - x[0]) / len(x)
    y = np.zeros((1, len(start_conditions)))
    y[0] = start_conditions

    for i in range(len(x) - 1):
        k1 = func(dv_type, x[i], y[i], *param_list)
        k2 = func(dv_type, x[i] + 0.5 * h, y[i] + 0.5 * h * k1, *param_list)
        k3 = func(dv_type, x[i] + 0.5 * h, y[i] + 0.5 * h * k2, *param_list)
        k4 = func(dv_type, x[i] + h, y[i] + h * k3, *param_list)

        y_new = y[i] + (h/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.append(y, [y_new], axis=0)
    return y



if __name__ == '__main__':
    v0 = 0
    phi0 = 1

    k = 0
    A = 0
    omega_f = 0

    L = 1
    g = 9.8
    omega = (g / L)**0.5

    a = 0
    b = 10
    N = 150
    t = np.linspace(a, b, N + 1)

    parameters = [omega, k, omega_f, A]


    # СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ МОДЕЛЕЙ

    y_nonlinear = rungeKutta(t, [v0, phi0], dv_nonlinear, parameters)
    y_linear = rungeKutta(t, [v0, phi0], dv_linear, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, y_nonlinear[:, 1], label="Нелинейная модель")
    axs[0].plot(t, y_linear[:, 1], label="Линейная модель")
    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"$\varphi$, рад")
    axs[0].grid()
    axs[0].legend()
    axs[1].scatter(y_nonlinear[:, 1], y_nonlinear[:, 0], marker='.', label="Нелинейная модель")
    axs[1].scatter(y_linear[:, 1], y_linear[:, 0], marker='.', label="Линейная модель")
    axs[1].set_xlabel(r"$\varphi$, рад")
    axs[1].set_ylabel(r"$v$, рад/с")
    axs[1].grid()
    axs[1].legend()
    plt.savefig("linear_nonlinear.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    # СРАВНЕНИЕ ЛИНЕЙНОЙ И НЕЛИНЕЙНОЙ МОДЕЛЕЙ ДЛЯ РАЗНЫХ НАЧАЛЬНЫХ УГЛОВ

    n_comparisons = 5
    phi0_list = np.linspace(0.125, 1, n_comparisons)

    y_nonlinear = [rungeKutta(t, [v0, phi0_list[i]], dv_nonlinear, parameters) for i in range(n_comparisons)]
    y_linear = [rungeKutta(t, [v0, phi0_list[i]], dv_linear, parameters) for i in range(n_comparisons)]
    legend_line_list = []
    fig, axs = plt.subplots(n_comparisons, 1, figsize=(6, 10))
    for i in range(n_comparisons):
        if i == 0:
            line1 = axs[i].plot(t, y_nonlinear[i][:, 1], label="Нелинейная модель")
            line2 = axs[i].plot(t, y_linear[i][:, 1], label="Линейная модель")
            legend_line_list = [line1, line2]
        else:
            axs[i].plot(t, y_nonlinear[i][:, 1])
            axs[i].plot(t, y_linear[i][:, 1])
        axs[i].set_xlabel(r"$t$, с")
        axs[i].set_ylabel(r"$\varphi$, рад")
        axs[i].grid()

        empty_line, = axs[i].plot([], [], ' ', label=rf"$\varphi_0 =$ {phi0_list[i]} рад")
        pseudo_title = axs[i].legend(handles=[empty_line], handlelength=0, handletextpad=0, loc='right')
        axs[i].add_artist(pseudo_title)

    fig.legend(handles=[l[0] for l in legend_line_list], loc='outside right lower')
    fig.savefig("lin_nonlin_4_phis.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    # СИЛА ТРЕНИЯ

    k = 0.2
    parameters = [omega, k, omega_f, A]
    y_nonlin = rungeKutta(t, [v0, phi0], dv_nonlinear, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, y_nonlin[:, 1], label=rf"$k = {k}$" + r"$c^{-1}$")
    axs[1].scatter(y_nonlin[:, 1], y_nonlin[:, 0], marker='.', label=rf"$k = {k}$" + r"$c^{-1}$")

    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"$\varphi$, рад")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel(r"$\varphi$, рад")
    axs[1].set_ylabel(r"$v$, рад/с")
    axs[1].grid()
    axs[1].legend()
    plt.savefig(f"with_k_{k}.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    a = 0
    b = 20
    N = 300
    t = np.linspace(a, b, N + 1)

    k = 0.6
    parameters = [omega, k, omega_f, A]
    y_nonlin = rungeKutta(t, [v0, phi0], dv_nonlinear, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, y_nonlin[:, 1], label=rf"$k = {k}$" + r"$c^{-1}$", color="orange")
    axs[1].scatter(y_nonlin[:, 1], y_nonlin[:, 0], marker='.', label=rf"$k = {k}$" + r"$c^{-1}$", color="orange")

    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"$\varphi$, рад")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel(r"$\varphi$, рад")
    axs[1].set_ylabel(r"$v$, рад/с")
    axs[1].grid()
    axs[1].legend()
    plt.savefig(f"with_k_{k}.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    # ВЫНУЖДАЮЩАЯ СИЛА

    a = 0
    b = 20
    N = 300
    t = np.linspace(a, b, N + 1)

    phi0 = 0.1

    k = 0.1
    A = 0.3
    omega_f = 5

    parameters = [omega, k, omega_f, A]
    y_linear = rungeKutta(t, [v0, phi0], dv_nonlinear, parameters)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t, y_linear[:, 1], color="violet")
    axs[1].scatter(y_linear[:, 1], y_linear[:, 0], marker='.', color="violet")
    axs[0].set_xlabel(r"$t$, с")
    axs[0].set_ylabel(r"$\varphi$, рад")
    axs[0].grid()
    axs[1].set_xlabel(r"$\varphi$, рад")
    axs[1].set_ylabel(r"$v$, рад/с")
    axs[1].grid()
    plt.savefig(f"driving_force.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    # БИЕНИЕ

    a = 0
    b_list = [60, 120]
    N = 300

    k = 0.05
    A = 1
    omega_f = 9
    parameters = [omega, k, omega_f, A]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(len(b_list)):
        t = np.linspace(a, b_list[i], N + 1)
        y_linear = rungeKutta(t, [v0, phi0], dv_nonlinear, parameters)

        axs[i, 0].plot(t, y_linear[:, 1], color="violet")
        axs[i, 1].scatter(y_linear[:, 1], y_linear[:, 0], marker='.', color="violet")

        axs[i, 0].set_xlabel(r"$t$, с")
        axs[i, 0].set_ylabel(r"$\varphi$, рад")
        axs[i, 0].grid()
        axs[i, 1].set_xlabel(r"$\varphi$, рад")
        axs[i, 1].set_ylabel(r"$v$, рад/с")
        axs[i, 1].grid()

    plt.savefig(f"beating.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


    # РЕЗОНАНС

    N_omegas_third = 7
    d = 0.3

    omega_f_left = np.linspace(omega - 2, omega - d, N_omegas_third)
    omega_f_center = np.linspace(omega - d, omega + d, N_omegas_third * 2)
    omega_f_right = np.linspace(omega + d, omega + 2, N_omegas_third)

    omega_f_list = np.concatenate([omega_f_left, omega_f_center, omega_f_right])
    omega_f_list = np.unique(omega_f_list)

    N_omegas = len(omega_f_list)

    a = 0
    b = 100
    N = 700
    t = np.linspace(a, b, N + 1)

    k = 0
    A = 0.5

    ampl_list = np.zeros(N_omegas)
    for i in range(N_omegas):
        parameters = [omega, k, omega_f_list[i], A]
        y_linear = rungeKutta(t, [v0, phi0], dv_linear, parameters)
        ampl_list[i] = max(abs(y_linear[:, 1]))

    plt.vlines(omega, 0, max(ampl_list) + 0.3, linestyles="dashed", color="orange", label=r"Резонанс, $\omega_f = \omega$")
    plt.plot(omega_f_list, ampl_list, label="Резонансная кривая")
    plt.xlabel(r"$\omega_f, рад/с$")
    plt.ylabel(r"$\varphi, рад$")
    plt.grid()
    plt.legend()
    plt.savefig("resonance.png", bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
