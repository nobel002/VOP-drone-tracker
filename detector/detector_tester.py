"""
Author: Emile Erauw

Description:
    This is the testing harnas for the drone classifier algorithm.
    This aids in testing how good the algorithm is at classifying sounds as either being drone sounds or
"""

from glob import glob
import numpy as np
import librosa
from scipy.spatial.distance import euclidean, cdist
import kmedoids
# from Simulation.drone_classifier import peak_detection, flatten_peaks, distance, band_filter
from detector import *

# Source - https://stackoverflow.com/a/68009718
# Posted by Prajot Kuvalekar, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-18, License - CC BY-SA 4.0
import warnings
warnings.filterwarnings('ignore')

drone_audio = glob("../../audio/Drone/*.wav")
other_audio = glob("../../audio/NoDrones/*.wav")


print(f"Found files: {drone_audio} and {other_audio}")

datapoints = min(len(drone_audio), len(other_audio), 1006)
drone_audio = drone_audio[:datapoints]
other_audio = other_audio[:datapoints]

# STFT Settings
nfft = 2**10
window_length = 2**10
hop = window_length // 4

# Setting this way higher ofcourse tanks performance
fmax = 1000 # Old value is 1000
k = 10 # Cluster count

def calc_droneness(file):
    """
    Yields a certainty metric determining the drone like characteristics of the audio.

    Parameters
    file (string): Path to the audio file.
    """
    y, fs = librosa.load(file, sr=None, mono=True)
    S = librosa.stft(y, n_fft=nfft, hop_length=hop, win_length=window_length)

    power_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    power_db = np.clip(power_db, -40, 0)

    freqs = librosa.fft_frequencies(sr=fs, n_fft=nfft)

    freq_ids = freqs <= fmax  # Limit the frequencies we're working with
    peaks = peak_detection(power_db, freq_ids)
    flattened_peaks, pc = flatten_peaks(peaks, freqs, fmax)

    flattened_peaks = flattened_peaks.reshape(-1, 1)
    distance_matrix = cdist(flattened_peaks, flattened_peaks, metric='cityblock')

    try:
        partitioning = kmedoids.fasterpam(distance_matrix, k)  # Partition into k clusters
        medoids = flattened_peaks[partitioning.medoids]
        bands = np.argwhere(medoids)[:, 0]

        return np.sum(power_db) / np.sum(band_filter(bands, power_db, 20, 1.5))
    except:
        print(f'No drone detected @ {file}')
        return -1000 # This means no drone (yeah I don't know what I'm doing anymore)

data = {
    'drone': [],
    'other': [],
}

for file in drone_audio:
    print(f"Processing {file}")
    data['drone'].append(calc_droneness(file))

for file in other_audio:
    print(f"Processing {file}")
    data['other'].append(calc_droneness(file))

# Pretty print the data

print("-"*40)
print(f"|{'Drone':^19}|{'Other':^19}|")
print("|" + "-"*39 + "|")

L = max(len(drone_audio), len(other_audio))

for i in range(L):
    drone, other = 0, 0
    if i < len(drone_audio):
        drone = data['drone'][i]
    if i < len(other_audio):
        other = data['other'][i]

    print(f"|{drone:^19.5f}|{other:^19.5f}|")

print("-"*40)
# drone_cut_off = 0.0065 # For the wavelet based transform

drone_cut_off = 18

print(f"Type I error: {sum(np.array(data['drone']) < drone_cut_off)*100/datapoints }%, Type II error: {sum(np.array(data['other']) > drone_cut_off)*100/datapoints}%")

