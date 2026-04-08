import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt

import kmedoids
from scipy.spatial.distance import (euclidean, cdist)
# Source - https://stackoverflow.com/a/68009718
# Posted by Prajot Kuvalekar, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-18, License - CC BY-SA 4.0
import warnings
warnings.filterwarnings('ignore')

def peak_detection(power_db, freq_ids):
    nT = power_db.shape[1]
    peaks = []

    for t in range(nT):
        deltaTime = power_db[freq_ids, t]

        if len(deltaTime) > 0:

            ####
            # Peak finding algo
            # ad hoc approach
            # This gives somewhat oke values...
            # With my amazing ad hoc optimization: Type I error: 15%, Type II error: 40%
            # p, _ = find_peaks(band, height=-20, distance=10)
            # Cut off was 0.0055

            # p = find_peaks_cwt(deltaTime, widths=5)

            p, _ = find_peaks(deltaTime, rel_height=10)

            peaks.append(p)
        else:
            peaks.append([])

    return peaks

def flatten_peaks(peaks, freqs, fmax):
    flattened_peaks = []

    for p in peaks:
        if len(p) > 0:
            flattened_peaks.extend(freqs[p])
    flattened_peaks = np.array(flattened_peaks)
    flattened_peaks = flattened_peaks[flattened_peaks < fmax]
    flattened_peaks.sort()
    peak_count = len(flattened_peaks)
    return flattened_peaks, peak_count

def band_filter(bands, input, depth=2, falloff=2.):
    """
    bands (Array): A list of indexes of frequency to apply this filter to
    input (Array Like): An input matrix to apply this filter to the STFT result
    depth (int): how much up and down do we need to include
    falloff (float): how much do we need to fall of. 1<= falloff <= +oo
    """
    filter = np.zeros_like(input)

    for band in bands:
        filter[band] = np.ones_like(input[band])
        for n in range(-depth, depth+1):
            if  0 <= band + n < len(input[band]):
                filter[band+n] += np.ones_like(input[band+n])*(falloff**(-abs(n)))

    return filter*input


if __name__ == "__main__":
    # I want custom pretty plots...
    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.sans-serif"] = ["Aptos"]
    plt.rcParams["image.cmap"] = "inferno"

    # Load audio
    filename = "../audio/Drone/B_S2_D1_067-bebop_000_.wav"
    y, sample_rate = librosa.load(filename, sr=None, mono=True)
    print("Sample Rate: ", sample_rate)

    # Spectrogram parameters
    nfft = 2**10
    hop = nfft // 4

    # Compute spectrogram
    S = librosa.stft(y, n_fft=nfft, hop_length=hop)
    power_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    power_db = np.clip(power_db, -40, 0)
    print(f"The power_db matrix is indexed as [f_bin, time_bin]")
    print(f"Shape of the stft matrix: {S.shape} == {power_db.shape} (=power_db M)")

    # Frequency axis
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=nfft)
    times = librosa.frames_to_time(np.arange(power_db.shape[1]), sr=sample_rate, hop_length=hop)
    print(f"freqs: {len(freqs)}, times: {len(times)}")
    # Limit the frequencies we're working with
    fmax = 4000
    freq_ids = freqs <= fmax # This is a mask so we only look at the specified frequencies


    # Plot spectrogram
    plt.figure()
    plt.imshow(power_db[freq_ids, :],
               cmap="inferno",
               origin="lower",
               aspect="auto",
               extent=(times[0], times[-1], freqs[freq_ids][0], freqs[freq_ids][-1]))

    plt.colorbar(label="dB")
    plt.title(f"Spectrogram - {filename[9:]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()
    plt.close()

    peaks = peak_detection(power_db, freq_ids)
    print(f"Peak count: {len(peaks)}")
    flattened_peaks, peak_count = flatten_peaks(peaks, freqs, fmax)

    # Distance matrix
    flattened_peaks = flattened_peaks.reshape(-1, 1)
    distance_matrix = cdist(flattened_peaks, flattened_peaks, metric='cityblock')
    # Okey
    distance_matrix = np.array(distance_matrix)
    # distance_matrix[distance_matrix > 250] = 1e6 # Penalize outliers

    # K-medoids clustering
    k = 10 # This number is way too arbitrary
    partitioning = kmedoids.fasterpam(distance_matrix, k) # Partition into k clusters
    medoids = flattened_peaks[partitioning.medoids]
    bands = np.argwhere(medoids)[:,0]
    # Plot medoids
    plt.figure()
    plt.imshow(
        power_db[freq_ids, :],
        aspect='auto',
        origin='lower',
        extent=(times[0], times[-1], freqs[freq_ids][0], freqs[freq_ids][-1])
    )

    for f in medoids:
        plt.axhline(f, color="green", linewidth=3)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram {filename[9:]}")
    plt.colorbar(label="dB")

    plt.show()

    # Filtering metric
    droneness =  np.sum(power_db) / np.sum(band_filter(bands, power_db, 20, 1.1))

    print("Droneness:", droneness)