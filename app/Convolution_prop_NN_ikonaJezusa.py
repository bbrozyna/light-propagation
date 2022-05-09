# -*- coding: utf-8 -*-
"""
Light propagation with convolution method

Paweł Komorowski
pawel.komorowski@wat.edu.pl
"""


import numpy as np
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras

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

def custom_weights_Re(shape, dtype=None):
    kernel = np.array([[1/(z*Lambda)*np.sin(np.pi*np.sqrt(x**2+y**2)**2/(z*Lambda)+2*np.pi*z/Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
    kernel = kernel.reshape(size, size, 1, 1)
    return kernel

def custom_weights_Im(shape, dtype=None):
    kernel = np.array([[-1/(z*Lambda)*np.cos(np.pi*np.sqrt(x**2+y**2)**2/(z*Lambda)+2*np.pi*z/Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
    kernel = kernel.reshape(size, size, 1, 1)
    return kernel

size = 128
c = 299792458
Nu = 140
Lambda = c/Nu*10**-6
Sigma = 20
f = 500
z = f
pixel = 1

# Arrays

# t = time.localtime()
# current_time = time.strftime("%H:%M:%S", t)
# print(current_time)

lens = np.array([[lens(np.sqrt(x**2+y**2),f,Lambda) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])
initAmp = np.array([[gaussian(np.sqrt(x**2+y**2),Sigma) for x in np.arange(-size/2,size/2)*pixel] for y in np.arange(-size/2,size/2)*pixel])


#field = np.array([np.real(initAmp*lens),np.imag(initAmp*lens)])
field = np.array([[[np.real(initAmp*lens)[i,j],np.imag(initAmp*lens)[i,j]] for i in range(size)] for j in range(size)])
field = field.reshape((1,size, size,2),order='F')
print(field.shape)

plt.imshow(field[:,:,0], interpolation='nearest')
plt.show()


plt.imshow(field[0,:,:,0], interpolation='nearest')
plt.show()



# Neural network

inputs = keras.Input(shape=(size,size,2))
x=keras.layers.Reshape((2,size,size))(inputs)

Re=keras.layers.Cropping2D(cropping=((1,0),0))(x)
Re=keras.layers.Reshape((size,size,1))(Re)
Im=keras.layers.Cropping2D(cropping=((0,1),0))(x)
Im=keras.layers.Reshape((size,size,1))(Im)

ReRe=Convolution2D(1,size,padding="same",kernel_initializer=custom_weights_Re,use_bias=False)(Re)
ImRe=Convolution2D(1,size,padding="same",kernel_initializer=custom_weights_Im,use_bias=False)(Re)
ReIm=Convolution2D(1,size,padding="same",kernel_initializer=custom_weights_Re,use_bias=False)(Im)
ImIm=Convolution2D(1,size,padding="same",kernel_initializer=custom_weights_Im,use_bias=False)(Im)

Re=keras.layers.Subtract()([ReRe,ImIm])
Im=keras.layers.Add()([ReIm,ImRe])
outputs=keras.layers.Concatenate(axis=-1)([Re,Im])

model=keras.Model(inputs=inputs, outputs=outputs)

# print(model.summary())

# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True,show_layer_names=False)

# Prpagation

conv=model(field)


# print(conv.numpy().shape)
# print(conv.numpy().reshape(size,size,2).shape)
# print(conv.numpy()[0][0][0])

#Visualisation

complexField=field.reshape(size,size,2)[:,:,0]+1j*field.reshape(size,size,2)[:,:,1]

plt.imshow(np.angle(complexField), interpolation='nearest')
plt.show()

plt.imshow(np.absolute(complexField), interpolation='nearest')
plt.show()

complexConv=conv.numpy().reshape(size,size,2)[:,:,0]+1j*conv.numpy().reshape(size,size,2)[:,:,1]

plt.imshow(np.angle(complexConv), interpolation='nearest')
plt.show()

