import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hamming  # Correct import for Hamming window
from collections import Counter

# ===============================
# Load external audio file (WAV)
# ===============================
Fs, x = wavfile.read(r"C:\Users\arman\phython plotting\input.wav")  # Replace with your WAV file path

# Convert stereo to mono
if x.ndim > 1:
    x = x[:, 0]

# Normalize signal
x = x / np.max(np.abs(x))

# ===============================
# Limit audio duration (1-30 s)
# ===============================
max_duration = 30  # seconds
x = x[:int(Fs * max_duration)]

# ===============================
# Band-pass filter: 80-500 Hz
# ===============================
def bandpass_filter(signal, Fs, lowcut=80, highcut=500, order=4):
    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

x = bandpass_filter(x, Fs)

# ===============================
# Frame parameters
# ===============================
frame_duration = 0.04  # 40 ms frames
N_frame = int(Fs * frame_duration)
hop_size = N_frame      # non-overlapping frames
num_frames = len(x) // hop_size

frame_pitches = []

# ===============================
# Frame-wise Autocorrelation pitch detection
# ===============================
for i in range(num_frames):
    x_frame = x[i*hop_size : i*hop_size + N_frame]

    # Apply Hamming window
    window = hamming(len(x_frame))
    x_frame = x_frame * window

    # Autocorrelation
    corr = np.correlate(x_frame, x_frame, mode='full')
    corr = corr[len(corr)//2:]  # Keep positive lags

    # Ignore zero-lag & find first significant peak
    d = np.diff(corr)
    pos_idx = np.where(d > 0)[0]

    if len(pos_idx) == 0:
        pitch = 0
    else:
        start_search = pos_idx[0]
        peak = np.argmax(corr[start_search:]) + start_search
        if peak != 0:
            pitch = Fs / peak
        else:
            pitch = 0

    # Keep only human voice pitches (80-500 Hz)
    if 80 <= pitch <= 500:
        frame_pitches.append(pitch)

frame_pitches = np.array(frame_pitches)

# ===============================
# Voice classification per frame
# ===============================
voice_types = []
for p in frame_pitches:
    if p == 0:
        continue
    elif 80 <= p < 160:
        voice_types.append("Male")
    elif 160 <= p < 250:
        voice_types.append("Female")
    else:
        voice_types.append("Child / Noise")

# ===============================
# Histogram visualization
# ===============================
plt.figure(figsize=(10, 6))
plt.hist(frame_pitches, bins=20, color='skyblue', edgecolor='black')
plt.title("Frame-wise Pitch Histogram (Autocorrelation)")
plt.xlabel("Pitch Frequency (Hz)")
plt.ylabel("Number of Frames")
plt.grid(True)
plt.show()

# ===============================
# Most dominant voice type
# ===============================
count = Counter(voice_types)
dominant_voice = count.most_common(1)[0][0]

# ===============================
# Output
# ===============================
print("Detected frame pitches (Hz):", np.round(frame_pitches, 1))
print("Frame-wise voice types:", voice_types)
print("Most dominant voice type in audio:", dominant_voice)

