# -*- coding: utf-8 -*-
"""
View image augmentations.
Modified code from:
    https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
"""


# example of horizontal shift image augmentation
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


#########################
# load the image
img = load_img('images//cats_vs_dogs//train//cat//cat.3.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
#########################

#########################
# Horizontal  Shift

# Shift is in pixel values, so this depends on size of image
datagen = ImageDataGenerator(width_shift_range=[-25,25])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
    # show the figure
pyplot.show()
#######################################


#######################################
# Vertical Shift

# Shift is in percentage of image
datagen = ImageDataGenerator(height_shift_range=0.2)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
    # show the figure
# show the figure
pyplot.show()
#######################################


#######################################
# Horizontal and Vertical Flip Augmentation
# True / False
# example of horizontal flip image augmentation
datagen = ImageDataGenerator(horizontal_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
# show the figure
pyplot.show()


datagen = ImageDataGenerator(vertical_flip=True)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
# show the figure
pyplot.show()
#######################################


#######################################
# Random rotation
# Degrees from 0 to 360

# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range=360)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
# show the figure
pyplot.show()
#######################################



#######################################
# Random Brightness
# Can either brighten or darken image (or both)
# Values less than 1.0 darken the image, e.g. [0.5, 1.0], 
# whereas values larger than 1.0 brighten the image, e.g. [1.0, 1.5], 
# where 1.0 has no effect on brightness.
# example of brighting image augmentation

datagen = ImageDataGenerator(brightness_range=[0.6,1.4])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
# show the figure
pyplot.show()
#######################################


#######################################
# Random zoom augmentation
# A zoom augmentation randomly zooms the image in and either 
# adds new pixel values around the image or interpolates pixel 
# values respectively.
datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.axis('off')
# show the figure
pyplot.show()
#######################################