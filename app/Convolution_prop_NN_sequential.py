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
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Constants and definitions

def h(r,z,l):
    return np.exp(1j*2*np.pi/(l*z))/(1j*l*z)*np.exp(1j*2*np.pi/(l*2*z)*r*r)

def gaussian(r,s):
    return np.exp(-r**2/(2*s**2))

def lens(r,f,l):
    return np.exp(1j*(-2*np.pi)/l*np.sqrt(r**2+f**2))

def custom_weights(shape, dtype=None):
    kernel = np.array([[h(np.sqrt(x**2+y**2),z,Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
    kernel = kernel.reshape(size, size, 1, 1)
    return kernel

size = 256
c = 299792458
Nu = 140
Lambda = c/Nu*10**-6
Sigma = 20
f = 500
z = f
pixel = 0.5

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
field = field.reshape(1,size, size,1)


# Propagation

model = Sequential()
 
model.add(Convolution2D(1, kernel_size=(size, size),padding="same",use_bias=False,kernel_initializer=custom_weights, input_shape=(size,size,1)))

conv=model(field)


t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

print(conv.numpy().shape)
print(conv.numpy()[0][0][0])
# Visualisation

plt.imshow(np.angle(field.reshape(size,size,1)), interpolation='nearest')
plt.show()

plt.imshow(np.absolute(field.reshape(size,size,1)), interpolation='nearest')
plt.show()


plt.imshow(np.absolute(conv.numpy().reshape(size,size,1)), interpolation='nearest')
plt.show()
