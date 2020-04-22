import wave
import pyaudio
from pylab import *
from scipy.io import wavfile
from scipy.fftpack import fft
from numpy.fft import fft, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter, lfilter
from scipy.optimize import least_squares,curve_fit

import glob

def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order,[low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def mic_function(gains, a,b,c,d,e):
    return a*np.exp(b*(gains**c) +d) + e

# fs_wanted = 1000.0
lowcut = 900
highcut = 1100



files_path = "/home/wael/Audio"
file_dir = glob.glob(files_path +"/*.wav")
audio_files_names = sort([x.split("/")[-1] for x in file_dir])
audio_files_distances = sort([float(x.replace("mic_calib_","").replace(".wav","")) for x in audio_files_names])
max_freq_list = []
for audio in audio_files_names:
    fs, data = wavfile.read(files_path + "/" + audio)
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order=3)
    total_time = len(data)/fs
    sp = fft(data)
    freq = fftfreq(len(data),1/fs)
    max_freq = max(np.abs(sp.real[:20000]))
    max_freq_list.append(max_freq)
# print(audio_files_distances)
# print(audio_files_names)
# print(max_freq_list)

# plt.plot(linspace(0,total_time,len(data)), data)
x0 = [0.1,-0.00005,1.,5.5,20]
popt, pcov = curve_fit(mic_function, max_freq_list, audio_files_distances, p0=x0)
print(popt)
a, b, c, d, e = popt
predicted = []
t = linspace(1000,60000,1000)


plt.scatter(max_freq_list,audio_files_distances, label="Raw Data")
plt.plot(t, mic_function(t,*popt),label = r"$a e^{bk^c +d} + f$")
plt.xlabel('Amplitude')
plt.ylabel("Distance (in)")
plt.grid(True)
plt.title("Condenser Microphone Calibration")
plt.legend()
plt.show()