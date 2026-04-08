import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, eval_legendre

c = 343
R = 0.1
theta = np.linspace(0, np.pi, 500)
phi = np.linspace(0, 2*np.pi, 500)


def hn(n,x):
    return spherical_jn(n,x) + 1j*spherical_yn(n,x)

def hnd(n,x):
    return spherical_jn(n,x,True) + 1j*spherical_yn(n,x,True)

def u(theta, omega):
    k = omega/c
    x = k*R
    ctheta = np.cos(theta)
    s = 0
    aantal = int(2*x + 10)
    for n in range(aantal):
        jnd = spherical_jn(n,x,True)
        term = -(2*n+1)*(1j**n)*(jnd/hnd(n,x))*hn(n,x)*eval_legendre(n,ctheta)
        s += term

    return s

def u_inc(theta,f):
    omega = 2*np.pi*f
    k = omega/c
    return np.exp(1j * k * R*np.cos(theta))

def plot_scattering(f):
    omega = 2*np.pi*f
    theta = np.linspace(0, 2*np.pi, 500)

    I = np.array([abs(u(t, omega)+u_inc(t,f))**2 for t in theta])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.plot(theta, I)

    # ax.set_title(f"Scattering patroon (f = {f}Hz)")

    plt.show()


def matrix_tabel(f,th=theta):  #opslaan!!!!!

    omega = 2*np.pi*f
    n = len(th)

    M = np.zeros((n,n))

    vals = np.array([(u(t, omega)+u_inc(t,f)).real for t in th])

    for i in range(n):
        for j in range(n):

            if abs(vals[j]) < 1e-12:
                M[i,j] = np.nan
            else:
                M[i,j] = (vals[i]/vals[j])

    return M

def heatmap(f,th=theta):

    M = matrix_tabel(f,th)

    plt.figure()

    plt.imshow(
        M,
        extent=[th[0], th[-1], th[-1], th[0]],
        aspect='auto'
    )

    plt.colorbar(label="u(θ_i)/u(θ_j)")
    plt.xlabel("θ_j")
    plt.ylabel("θ_i")
    # plt.title(f"Heatmap scattering verhouding (f = {f}Hz)")

    plt.show()


def hoek(phi1, th1, phi2, th2):
    alpha = np.arccos(np.sin(phi1)*np.sin(phi2)+np.cos(phi1)*np.cos(phi2)*np.cos(th1-th2))
    return alpha

def richting_multifreq_4mics(u_list, mic_pos, freq, th=theta):

    n_mics = len(u_list)

    dtheta = th[1] - th[0]

    tol_angle = np.deg2rad(2)
    k_tol = int(round(tol_angle / dtheta))

    best_theta = None
    best_error = np.inf

    # precompute alle hoekverschillen tussen mics
    psi = np.zeros((n_mics, n_mics))

    for i in range(n_mics):
        for j in range(n_mics):
            phi1, th1 = mic_pos[i]
            phi2, th2 = mic_pos[j]

            psi[i, j] = hoek(phi1, th1, phi2, th2)

    # scan mogelijke richtingen
    for i in range(len(th)):

        total_error = 0

        for f_idx, f in enumerate(freq):

            omega = 2*np.pi*f
            vals = np.array([u(t, omega) for t in th])

            # alle microfoonparen
            for m1 in range(n_mics):
                for m2 in range(m1+1, n_mics):

                    u1 = u_list[m1][f_idx]
                    u2 = u_list[m2][f_idx]

                    if abs(u2) < 1e-12:
                        continue

                    psi_ij = psi[m1, m2]
                    k = int(round(psi_ij / dtheta))

                    meas_ratio = abs(u1)/abs(u2)

                    local_best = np.inf

                    for dk in range(-k_tol, k_tol+1):

                        j = i + k + dk

                        if 0 <= j < len(th):

                            if abs(vals[j]) > 1e-12:

                                model_ratio = abs(vals[i])/abs(vals[j])

                                err = abs(model_ratio - meas_ratio)

                                if err < local_best:
                                    local_best = err

                    total_error += local_best

        if total_error < best_error:

            best_error = total_error
            best_theta = th[i]

    return best_theta, best_error
