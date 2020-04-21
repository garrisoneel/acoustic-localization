import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as image
import librosa
import librosa.display
import os
import math
import csv

samplesdir = os.path.abspath("./data")
speed_of_sound = 343.0
x_size = 65.5*0.0254
y_size = 30.5*0.0254
resolution = 0.01

max_dist = np.sqrt(x_size*x_size + y_size * y_size)

x_shape = int(x_size/resolution)
y_shape = int(y_size/resolution)

predict_sigma = 10

update_sigma = 5

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

def transform_pt(x,y,mic1,mic2):
    theta = np.arctan2(mic2.y - mic1.y, mic2.x - mic1.x)
    theta_pt = np.arctan2(y,x)
    r = np.sqrt(x*x + y*y)

    x_out = r * np.cos(theta + theta_pt) + mic1.x
    y_out = r * np.sin(theta + theta_pt) + mic1.y

    return [x_out,y_out]

def pt_to_index(x,y):

    if(math.isnan(x)):
        return None

    if(math.isnan(y)):
        return None

    x_out = int(x/resolution)
    y_out = int(y/resolution)

    if(x_out < 0 or x_out >= x_shape):
        return None

    if(y_out < 0 or y_out >= y_shape):
        return None

    return [x_out , y_out]

def circ_intersection(d,R1,R2):
    d2 = d*d
    R22 = R2*R2
    R12 = R1*R1

    x = (d2 - R22 + R12)/(2.0*d)
    a = 1.0/d * np.sqrt(4*d2*R12 - (d2 - R22 + R12) * (d2 - R22 + R12) )

    return x,a

def get_curve(mic1,mic2,delta):

    if(delta == None):
        print("bad delta")
        return None
        

    layer = np.zeros([x_shape,y_shape])

    delta_dist = delta*speed_of_sound

    d = math.sqrt( (mic1.x - mic2.x)*(mic1.x - mic2.x) + (mic1.y - mic2.y)*(mic1.y - mic2.y) )

    if(abs(delta_dist) > d):
        print(d)
        print(delta_dist)
        print(delta)
        return None

    r0 = (d - delta_dist)*0.5
    c = 0
    for i in range(0,10000):

        r1 = max_dist*i/10000.0 + r0
        r2 = max_dist*i/10000.0 + delta_dist + r0

        x,a = circ_intersection(d,r1,r2)
    
        x3 = [x,x]
        y3 = [0.5*a, -0.5*a]

        pt_t = transform_pt(x3[0],y3[0],mic1,mic2)

        out_pt = pt_to_index(pt_t[0],pt_t[1])
        if(out_pt != None):
            layer[out_pt[0],out_pt[1]] = 1.0

        pt_t = transform_pt(x3[1],y3[1],mic1,mic2)

        out_pt = pt_to_index(pt_t[0],pt_t[1])
        if(out_pt != None):
            layer[out_pt[0],out_pt[1]] = 1.0
            c += 1

    if(np.sum(layer)==0):
        print("no curve")
        return None
    return layer + 0.0001*np.ones([x_shape,y_shape])/(x_shape*y_shape)

def predict(old_field):
    old_field = image.gaussian_filter(old_field, predict_sigma)
    s = np.sum(old_field)
    if s > 0 :
        old_field /= s
    return  old_field

def get_delta(i,time,m1_index,m2_index):

    t1_index = None
    t1_min = 10.0

    for j in range(max(0,i-20), min(len(data[m1_index]), i+21) ):
        ref_min = abs(data[m1_index][j] - time)
        if(ref_min < t1_min):
            t1_min = ref_min
            t1_index = j

    #print(t1_min)
    if(t1_min > 0.0125):
        return None

    t2_index = None
    t2_min = 10.0
    
    for j in range(max(0,i-40), min(len(data[m2_index]),i+41) ):
        ref_min = abs(data[m2_index][j] - time)
        if(ref_min < t2_min):
            t2_min = ref_min
            t2_index = j

    #print(t2_min)
    if(t2_min > 0.0125):
        return None

    return data[m2_index][t2_index] - data[m1_index][t1_index]

def get_delta_from_delays(i,m1_index,m2_index):
    d1 = 0.0
    d2 = 0.0

    if(m1_index != 0):
        d1 = delays[m1_index][i]
    if(m2_index != 0):
        d2 = delays[m2_index][i]

    #print(d1)
    #print(d2)

    return d2-d1

def update(old_field, i):
    delta = 0.5
    #time = data[0][i]

    for j in range(0,len(mics)):
        for k in range(j+1,len(mics)):
        

            m1 = mics[j]
            m2 = mics[k]

            #print("j = %s" % j)
            #print("k = %s" % k)

            #delta = get_delta(i,time,m1.index,m2.index)
            delta = get_delta_from_delays(i,m1.index,m2.index)
            
            curve = get_curve(m1,m2, delta)
            if(not(curve is None)):
                out_img = curve
                out_img = image.gaussian_filter(out_img, update_sigma)
                old_field *= out_img
                #plt.figure()
                #plt.imshow(np.transpose(out_img),origin = 'lower',interpolation='none')
                #plt.savefig(str(i) + '_' + str(j) +"-"+ str(k)+ '.png', bbox_inches='tight')
                #plt.close()

    s = np.sum(old_field)
    if s > 0 :
        old_field /= s
    return  old_field

def read_data():
    print("Using samples from %s" % (samplesdir))
    
    plt.figure()

    colors = [[1,0,0],[0,1,0],[0,0,1],[1,0,1]]
    i = 0
    for file in os.listdir(samplesdir):
        if file.endswith('.wav'):
            print("reading %s" % (file))
            x, sr = librosa.load(os.path.join(samplesdir,file))  
            print(sr)          
            #plt.figure()
            #librosa.display.waveplot(x,sr=sr)          
            onset_frames = librosa.onset.onset_detect(x,sr=sr,units='samples',hop_length=16)
            #plt.scatter(onset_frames,0.1 * i * np.ones(len(onset_frames)), c=colors[i])
            
            onset_times = librosa.samples_to_time(onset_frames,sr=sr)
            #plt.scatter(onset_times,np.ones(len(onset_times)))
            print(len(onset_times))
            data.append(onset_times)

            i += 1

    with open('onset_time.csv', mode='w') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(data[0])
        file.writerow(data[1])
        file.writerow(data[2])
        file.writerow(data[3])

    #plt.show()

def read_data_from_file():
    with open('onset_time.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in  csv_reader:
            data.append([float(i) for i in row])
    pass

def read_delays_from_file():
    with open('delays.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in  csv_reader:
            delays.append([float(i) for i in row])


def main():

    #read_data()
    #read_data_from_file()
    read_delays_from_file()

    field = np.ones([x_shape,y_shape])/(x_shape*y_shape)

    for i in range(0,len(delays[0])):
        
        #print(get_delta(i,data[0][i],0,2))

        field = update(field, i)

        #field += 0.0001*np.ones([x_shape,y_shape])/(x_shape*y_shape)
    #plt.imshow(np.transpose(field),origin = 'lower',interpolation='none')
    #plt.show()
        print("--------")
        print(str(i) + " / " + str(len(delays[0]) - 1))
        print("--------")
        field = predict(field)
        plt.figure()
        plt.imshow(np.transpose(field),origin = 'lower',interpolation='none')
        plt.savefig('./images/a' + str(i) + '.png', bbox_inches='tight')
        plt.close()
    
    #plt.show()

#curve = get_curve(mics[1],mics[2],-0.001)
#if curve is None:
#    print("None")
#    quit()
#plt.figure()
#plt.imshow(np.transpose(curve),origin = 'lower',interpolation='none')
#plt.show()


main()