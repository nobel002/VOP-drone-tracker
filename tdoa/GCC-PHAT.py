import numpy as np
import matplotlib.pyplot as plt

a = np.load('audio2.npy')

SAMPLERATE = len(a[:,0])
mic1 = a[:,0]
mic2 = a[:,1]
mic3 = a[:,2]
mic4 = a[:,3]

def calculate_fft(signal, samplerate=SAMPLERATE):

    n = len(signal)
    # Apply a Hanning window to smooth the edges of the signal
    window = np.hanning(n)

    sig_windowed = signal * window

    # Perform FFT
    freq_data = np.fft.fft(sig_windowed, axis=0)

    # Calculate frequency bins
    freq_bins = np.fft.fftfreq(n, 1 / samplerate)

    return freq_bins, freq_data

def calculate_ifft(freq_bins, freq_data, samplerate=SAMPLERATE):
    n = len(freq_bins) * 2 - 2  # Recover original signal length

    # Reconstruct the Hanning window used in the forward FFT
    window = np.hanning(n)

    # Perform inverse FFT
    signal = np.fft.ifft(freq_data, n=n, axis=0)

    # Compensate for the Hanning window
    # Avoid division by zero for near-zero window values
    window_safe = np.where(window > 1e-10, window, 1e-10)
    signal = signal / window_safe

    return signal

bin1, X1 = calculate_fft(mic3, SAMPLERATE)
bin2, X2 = calculate_fft(mic1, SAMPLERATE)

G = X1*np.conjugate(X2)

R = calculate_ifft(bin1,G)
tau = np.argmax(R)

print(tau/SAMPLERATE)
plt.plot(((np.abs(R[:]))))
plt.show()
