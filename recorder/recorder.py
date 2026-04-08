import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# --- Instellingen ---
SAMPLERATE = 44100  # Samples per seconde
WINDOW_MS = 200     # Hoeveel milliseconden we tegelijkertijd tonen
CHANNELS = 2
# wss kan je gewoon Channels =4 nemen voor de vier microfoons.
print(sd.query_devices())
DEVICE1 = 15       #selecteer device
DEVICE2 = 16

print('geeft theta in:')
theta = float(input())
print('geeft phi in:')
phi = float(input())
filename = f"audio_theta_{theta}_phi_{phi}.npy"

# Maak een buffer aan om de inkomende data in op te slaan
audio_data1 = np.zeros((SAMPLERATE, CHANNELS))
audio_data2 = np.zeros((SAMPLERATE, CHANNELS))


def audio_callback1(indata, frames, time, status):
    """Deze functie vult onze audio_data buffer aan."""
    global audio_data1
    # Schuif de oude data op en voeg de nieuwe toe (rolling buffer)
    shift = len(indata)
    audio_data1 = np.roll(audio_data1, -shift, axis=0)
    audio_data1[-shift:, :] = indata

def audio_callback2(indata, frames, time, status):
    """Deze functie vult onze audio_data buffer aan."""
    global audio_data2
    # Schuif de oude data op en voeg de nieuwe toe (rolling buffer)
    shift = len(indata)
    audio_data2 = np.roll(audio_data2, -shift, axis=0)
    audio_data2[-shift:, :] = indata

# def calculate_fft(signal, samplerate=SAMPLERATE):

#     n = len(signal)
#     # Apply a Hanning window to smooth the edges of the signal
#     window = np.hanning(n)


#     sig_windowed = signal * window

#     # Perform FFT
#     freq_data = np.fft.rfft(sig_windowed, axis=0)
#     # Get the magnitude (absolute value)
#     magnitude = np.abs(freq_data)

#     # Calculate frequency bins
#     freq_bins = np.fft.rfftfreq(n, 1 / samplerate)

#     return freq_bins, magnitude


# Start opname
try:
    # Open de microfoon stream
    stream1 = sd.InputStream(channels=CHANNELS, samplerate=SAMPLERATE, callback=audio_callback1,device=DEVICE1)
    stream2 = sd.InputStream(channels=CHANNELS, samplerate=SAMPLERATE, callback=audio_callback2,device=DEVICE2)

    with stream1,stream2:
        sd.sleep(1000)
        audio_data = np.hstack((audio_data1,audio_data2))
        fig,ax = plt.subplots(4)
        np.save(filename.replace(".", "-"), audio_data)


except Exception as e:
    print(f"Fout: {e}")

