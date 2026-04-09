import numpy as np
from scipy.optimize import least_squares
import GCC_PHAT as TDOA
from math import sin, cos, pi
from geometrie import *
from itertools import combinations, permutations
data = np.load('../recorder/audio_theta_90.0_phi_180.0.npy')
mics = mic_array
SAMPLERATE = 44100
c = 343 # m/s

def get_angles(data=data):
    stelsel = []

    for i, j in combinations([0, 1, 2, 3], 2):
        TDij = TDOA.get_TDOA(data[:, i], data[:, j], SAMPLERATE)
        TDji = TDOA.get_TDOA(data[:, j], data[:, i], SAMPLERATE)

        tau = min(TDij, TDji)

        dx = deltax(mics[i], mics[j]) if TDij < TDji else deltax(mics[j], mics[i])
        dy = deltay(mics[i], mics[j]) if TDij < TDji else deltay(mics[j], mics[i])
        dz = deltaz(mics[i], mics[j]) if TDij < TDji else deltaz(mics[j], mics[i])

        stelsel.append(
            lambda theta, phi: dx * sin(theta) * cos(phi) + dy * sin(theta) * sin(phi) + dz * cos(theta) - tau * c)

    def score(angles):
        return sum([vgl(angles[0], angles[1]) * vgl(angles[0], angles[1]) for vgl in stelsel])

    res = least_squares(score, (0, 0), bounds=([-pi, -2 * pi], [pi, 2 * pi]))
    if res.success:
        theta, phi = res.x
        if theta < 0:
            theta *= -1
            phi += pi

        print(f"theta: {theta}, phi: {phi}")
        return theta, phi
    else:
        print("Couldn't find a solution :,(")
        return -1
