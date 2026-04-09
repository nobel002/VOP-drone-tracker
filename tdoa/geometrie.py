import numpy as np

#
# mic1 =
# mic2 =
# mic3 =
# mic4 =
#

mic_array = [
    np.array([0.1, np.pi / 3, np.pi / 3]),
    np.array([0.1, np.pi / 3, np.pi]),
    np.array([0.1, np.pi / 3, 5 * np.pi / 3]),
    np.array([0.1, np.pi, 0])
]


def x(r, theta, phi):
    return r * np.sin(theta) * np.cos(phi)


def y(r, theta, phi):
    return r * np.sin(theta) * np.sin(phi)


def z(r, theta, phi):
    return r * np.cos(theta)


def deltax(m1, m2):
    return x(m1[0], m1[1], m1[2]) - x(m2[0], m2[1], m2[2])


def deltay(m1, m2):
    return y(m1[0], m1[1], m1[2]) - y(m2[0], m2[1], m2[2])


def deltaz(m1, m2):
    return z(m1[0], m1[1], m1[2]) - z(m2[0], m2[1], m2[2])


