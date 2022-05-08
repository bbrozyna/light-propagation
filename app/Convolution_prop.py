# -*- coding: utf-8 -*-
"""
Light propagation with convolution method

Paweł Komorowski
pawel.komorowski@wat.edu.pl
"""


import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import time



# Constants and definitions

def h(r,z,l):
    return np.exp(1j*2*np.pi/(l*z))/(1j*l*z)*np.exp(1j*2*np.pi/(l*2*z)*r*r)

def gaussian(r,s):
    return np.exp(-r**2/(2*s**2))

def lens(r,f,l):
    return np.exp(1j*(-2*np.pi)/l*np.sqrt(r**2+f**2))

size = 256
c = 299792458
Nu = 140
Lambda = c/Nu*10**-6
Sigma = 20
f = 500
z = f
pixel = 1

# Arrays

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

hkernel = np.array([[h(np.sqrt(x**2+y**2),z,Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
lens = np.array([[lens(np.sqrt(x**2+y**2),f,Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
initAmp = np.array([[gaussian(np.sqrt(x**2+y**2),Sigma) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

field=initAmp*lens


# Propagation

conv = signal.fftconvolve(field, hkernel, mode='same')

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)


# Visualisation

#faza
plt.imshow(np.angle(field), interpolation='nearest')
plt.show()

#amplituda
plt.imshow(np.absolute(field), interpolation='nearest')
plt.show()

#amplituda wyniku
plt.imshow(np.absolute(conv), interpolation='nearest')
plt.show()
