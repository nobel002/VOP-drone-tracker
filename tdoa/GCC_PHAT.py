import numpy as np
import matplotlib.pyplot as plt

a = np.load('../recorder/audio_theta_90.0_phi_180.0_new.npy')

SAMPLERATE = 44100
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
    freq_data = np.fft.rfft(sig_windowed,n, axis=0)

    # Calculate frequency bins
    freq_bins = np.fft.rfftfreq(n, 1 / samplerate)

    return freq_bins, freq_data

def calculate_ifft(freq_bins, freq_data, samplerate=SAMPLERATE):
    n = len(freq_data)  # Recover original signal length

    # Reconstruct the Hanning window used in the forward FFT
    # window = np.hanning(n)

    # Perform inverse FFT
    signal = np.fft.irfft(freq_data, n=n, axis=0)

    # Compensate for the Hanning window
    # Avoid division by zero for near-zero window values
    # window_safe = np.where(window > 1e-10, window, 1e-10)
    # signal = signal / window_safe

    return signal


def get_TDOA(mic1, mic2, SAMPLERATE=SAMPLERATE):

    # mic1 *= np.hanning(len(mic1))
    # mic2 *= np.hanning(len(mic2))

    X1 = np.fft.fft(mic1, SAMPLERATE)
    X2 = np.fft.fft(mic3, SAMPLERATE)

    G = X1*np.conjugate(X2)
    G/=np.linalg.norm(G)

    R = np.fft.ifft(G)
    tau = np.argmax(R)

    print(tau/SAMPLERATE)
    plt.plot(np.abs(R[:]))
    plt.show()
    return tau/SAMPLERATE


get_TDOA(mic3, mic2, SAMPLERATE)


