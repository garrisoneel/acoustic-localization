import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as image
import librosa
import librosa.display
import os
import math
import csv

speed_of_sound = 343.0
x_size = 65.5*0.0254
y_size = 30.5*0.0254
resolution = 0.01

max_dist = np.sqrt(x_size*x_size + y_size * y_size)

data = []
delays = []

class mic:
    x = 0
    y = 0
    index = None
    def __init__(self,x,y,index):
        self.x = x
        self.y = y
        self.index = index

mics = [mic(0,y_size,2),mic(0,0,0),mic(x_size,y_size,1),mic(x_size,0,3)]

for j in range(0,len(mics)):
        data.append([])
        delays.append([])

for i in range(0,500):
    x = 0.5*y_size*math.cos(i*math.pi*2/100.0) + x_size*0.5
    y = 0.5*y_size*math.sin(i*math.pi*2/100.0) + y_size*0.5

    t = 0.25*i
    for j in range(0,len(mics)):
        d = math.sqrt( (x-mics[j].x)*(x-mics[j].x) + (y-mics[j].y)*(y-mics[j].y) )
        data[mics[j].index].append(t + d/speed_of_sound)
        #print(d)

    for j in range(0,len(mics)):
        if(mics[j].index == 0):
            delays[0].append(data[0][i])
        else:
            delays[mics[j].index].append(data[mics[j].index][i] - data[0][i])
            
    
with open('delays_sim.csv', mode='w') as file:
    file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(delays)):
        file.writerow(delays[i])

    