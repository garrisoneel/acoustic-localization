#%%
from scipy.io.wavfile import read as wavread
from scipy import signal
import numpy as np
import os
from matplotlib import pyplot as plt

samplesdir = os.path.abspath("./samples/1")
print("Using samples from %s" % (samplesdir))
# samples = [file for file in os.listdir(samplesdir) if file.endswith('.wav')]
# print(samples)
# sample_file = samples[0]
data = []
for file in os.listdir(samplesdir):
    if file.endswith('.wav'):
        (sr,dat) = wavread(os.path.join(samplesdir,file))
        data.append(dat[:,0])

# (sample_rate,sample1) = wavread(os.path.join(samplesdir,sample_file))
# print(sample_rate, data.shape)
# fig1 = plt.figure()
# plt.plot(data[0])
# plt.plot(data[1])
# plt.title("before")
# # plt.show()
d0 = abs(np.gradient(np.gradient(data[0][1000:]/np.max(data[0]))))
d1 = abs(np.gradient(np.gradient(data[1][1000:]/np.max(data[1]))))
thresh = 0.01
d0[d0<thresh] = 0
d1[d1<thresh] = 0

offs = np.argmax(signal.correlate(d0,d1)) - len(d1)
print(offs)

# plt.figure()
# plt.plot(signal.correlate(data[0],data[1]))

fig2 = plt.figure()
plt.plot(d0)
plt.plot(d1)
plt.title("gradients")

plt.figure()
plt.plot(np.arange(len(data[1])) + offs, data[1])
plt.plot(data[0])
plt.title("aligned by correllating gradients")

# plt.show()
#%%
from scipy.signal import find_peaks, find_peaks_cwt
ref = data[0]
# for ref in data:
hp_filt = signal.butter(4, 18000, 'high', fs=44100,analog=False, output='sos')
filt_ref = signal.sosfilt(hp_filt, ref)
mx = np.max(filt_ref)
# ref = ref/mx
(pks_ref,props) = find_peaks(abs(filt_ref), distance=1000, height=mx/5, wlen=100)
# print(pks_ref)
plt.figure()
plt.title("finding peaks")
plt.plot(filt_ref)
plt.plot(pks_ref,filt_ref[pks_ref],'*')

#%%
fig3 = plt.figure()
plt.title("aligned using peaks")
offsets = []
for mic in data:
    hp_filt = signal.butter(4, 18000, 'high', fs=44100,analog=False, output='sos')
    filt_mic = signal.sosfilt(hp_filt, mic)
    mx = np.max(filt_mic)
    (pks,props) = find_peaks(abs(filt_mic), distance=1000, height=mx/5, wlen=100)
    # print(pks)
    print(pks_ref[-2:]-pks[-2:], np.mean(pks_ref[-2:]-pks[-2:]))
    offset = int(np.mean(pks_ref[-2:]-pks[-2:]))
    plt.plot(np.arange(len(mic))+offset, mic)
plt.plot(ref)
plt.show()

