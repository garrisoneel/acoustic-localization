import wave
import pyaudio
from pylab import *
from scipy.io import wavfile
from numpy.fft import fft, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt

files_path = "/home/wael/Audio/mic_calib_20.wav"
fs, data = wavfile.read(files_path)
total_time = len(data)/fs
time = np.linspace(0,total_time,len(data))
sp = fft(data)
freq = fftfreq(len(data),1/fs)
print(max(np.abs(sp.real[:20000])))
plt.plot(freq[:20000],np.abs(sp.real[:20000])) 
plt.show()