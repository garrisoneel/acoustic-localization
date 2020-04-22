import wave
import pyaudio
from pylab import *
from scipy.io import wavfile
from numpy.fft import fft, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import glob

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




files_path = "/home/wael/Audio"
file_dir = glob.glob(files_path +"/*.wav")
audio_files_names = sort(np.array([x.split("/")[-1] for x in file_dir]))
audio_files_distances = sort(np.array([float(x.replace("mic_calib_","").replace(".wav","")) for x in audio_files_names]))
max_freq_list = []
for audio in audio_files_names:
    fs, data = wavfile.read(files_path + "/" + audio)
    total_time = len(data)/fs
    sp = fft(data)
    freq = fftfreq(len(data),1/fs)
    max_freq = max(np.abs(sp.real[:20000]))
    max_freq_list.append(max_freq)
print(audio_files_distances)
print(audio_files_names)
print(max_freq_list)


plt.scatter(audio_files_distances,max_freq_list) 
plt.show()