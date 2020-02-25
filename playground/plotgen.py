# import matplotlib as mpl
# mpl.use('agg')
#%% 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def gauss(mu=0, stdev=1):
    return lambda x: 1/(stdev*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/stdev)**2)


def toPolar(x, y): return (np.sqrt(x*x+y*y), np.arctan2(y, x))


def toCart(r,phi): return (r*np.cos(phi), r*np.sin(phi))


def heart(phi): return 0.5*(1+np.cos(phi))


def pdf_theta(theta):
    return gauss(heart(theta), 0.1 * abs(theta)/np.pi/2.0+0.21)


def pdf_rad(r):
    return lambda phi: gauss(heart(phi), 0.3 * abs(phi)/np.pi/2.0+0.01)(r)

def Atod(A):
    return np.sqrt(5/A) #also made up

def dtoA(d):
    return 5/d**2 #completely made up

def xytoA(x,y):
    (d,theta) = toPolar(x,y)
    return dtoA(d)*heart(theta)

#%% Build & plot a PDF

# numpts = 30
# x = np.linspace(-0.5,1.5,numpts)
# y = np.linspace(-2,2,numpts*2)
# X,Y = np.meshgrid(x,y)

# (ds,angles) = toPolar(X,Y)
# ds = np.reshape(ds,(ds.size,1))
# angles = np.reshape(angles,(angles.size,1))
# # print(ds, angles)
# vals = np.zeros(ds.shape)
# for ix, (fun, rd) in enumerate(zip(map(pdf_theta,angles),ds)):
#     vals[ix] = fun(rd)
# # print("x: {xs}, y: {ys}, z: {zs}".format(xs=x.shape,ys=y.shape,zs=vals.shape))
# Z = np.reshape(vals,X.shape)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X,Y,Z,cmap='cubehelix')
# fig.savefig('temp.png')


#%% find "amplitude" at each microphone
robo_pos = np.array([3,0])
mic_poses = np.array([[0,0],[0,5],[0,-5]])
rel = np.array([robo_pos-mic_poses[p][:] for p in range(mic_poses.shape[0])])
A = xytoA(rel[:,0], rel[:,1])

for k in range(3):
    print(rel[k,0], rel[k,1])

#%% now build probability distributions
dists = Atod(A)
print(dists)
x = np.linspace(0,10,100)
y = np.linspace(-10,10,100)
X,Y = np.meshgrid(x,y)
print(X.shape, Y.shape)
pbj = np.zeros(X.shape)
individual = [np.zeros(X.shape) for k in range(3)]
fn = []
for mic in range(3):
    pdf = gauss(dists[mic],0.5+0.1*dists[mic])
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            xt = xx + mic_poses[mic,0]
            yt = yy + mic_poses[mic,1]

            d,th = toPolar(xt,yt)
            d_real = d/heart(th)
            individual[mic][iy,ix] = pdf(d_real)
            pbj[iy,ix] += pdf(d_real)

fig = plt.figure()

ax = fig.add_subplot(232, projection='3d')
ax2 = fig.add_subplot(234, projection='3d')
ax3 = fig.add_subplot(235, projection='3d')
ax4 = fig.add_subplot(236, projection='3d')

ax.set_autoscalez_on(False)
ax.set_zmargin(1)
ax.plot_surface(X,Y,pbj/3,cmap='cubehelix', linewidth=0, antialiased=False, rstride=1, cstride=1)#, rcount=200, ccount=200)
ax2.set_autoscalez_on(False)
ax2.set_zmargin(1)
ax2.plot_surface(X,Y,individual[0],cmap='cubehelix', linewidth=0, antialiased=False, rstride=1, cstride=1)#,  rcount=200, ccount=200)
ax3.set_autoscalez_on(False)
ax3.set_zmargin(1)
ax3.plot_surface(X,Y,individual[1],cmap='cubehelix', linewidth=0, antialiased=False, rstride=1, cstride=1)#, cstride=1, rcount=200, ccount=200)
ax4.set_autoscalez_on(False)
ax4.set_zmargin(1)
ax4.plot_surface(X,Y,individual[2],cmap='cubehelix', linewidth=0, antialiased=False, rstride=1, cstride=1)#, cstride=1, rcount=200, ccount=200)



idx_max = np.unravel_index(np.argmax(pbj, axis=None), pbj.shape)
pmax = pbj[idx_max[0],idx_max[1]]/3
xmax = X[idx_max[0],idx_max[1]]
ymax = Y[idx_max[0],idx_max[1]]
print("Max P ({}) at X: {:.2f}, Y: {:.2f}".format(pmax,xmax,ymax))
plt.show()
fig.savefig('3-mics.png')
