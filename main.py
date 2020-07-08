# -*- coding: utf-8 -*-
"""
@author: EJMorales
"""


import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pathlib
import time
import image_handler as ih
import numpy as np
import pandas as pd

##########################
START = 0
def time_start():
    global START 
    START = time.perf_counter()
    
def time_end():
    global START
    end = time.perf_counter() - START
    
    print(time.strftime('%H:%M:%S', time.gmtime(end)))
##########################

##########################
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.995):
      print("\nReached 99.5% accuracy, stopping training.")
      self.model.stop_training = True
stop_training = myCallback()
##########################

if __name__ == "__main__":
    image_width = 128
    image_height = 128
    image_depth = 3
    input_shape = (image_width, image_height, image_depth)
    '''
    ##################
    # Move dogs to respective folders
    raw_images = glob.glob(str(pathlib.Path('images/raw/train/dog*')))
    print('Dogs: ', len(raw_images), sep=' : ')
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(mpimg.imread(raw_images[i]) )
        plt.axis('off')
    plt.show()
    
    time_start()
    ih.copy_images(raw_images, 'images/dogs', resize=None)
    time_end()
    ##################
    
    ##################
    # Move cats to respective folders
    raw_images = glob.glob(str(pathlib.Path('images/raw/train/cat')))
    print('Cats: ', len(raw_images), sep=' : ')
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(mpimg.imread(raw_images[i]) )
        plt.axis('off')
    plt.show()
    
    time_start()
    ih.copy_images(raw_images, 'images/cats', resize=None)
    time_end()
    ##################
    '''
    
    ##################
    # Check dimension range of images
    image_list = glob.glob(str(pathlib.Path('images/raw/train/*')))
    time_start()
    image_dimensions = ih.get_image_dimensions(image_list)
    time_end()
    
    
    # View histogram of widths and heights 
    data = image_dimensions[:,0]
    bins = np.arange(100, 500, 5) # fixed bin size
    plt.xlim([100, 500+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('')
    plt.xlabel('Image Width (bin size = 5)')
    plt.ylabel('Frequency')
    plt.show()
    
    data = image_dimensions[:,1]
    bins = np.arange(100, 500, 5) # fixed bin size
    plt.xlim([100, 500+5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('')
    plt.xlabel('Image Height (bin size = 5)')
    plt.ylabel('Frequency')
    plt.show()
    ##################
    
    ##################
    '''
    Structure Images for ImageDataGenerator
    dogs_vs_cats
    ├── train
    │ ├── cats
    │ └── dogs
    ├── val
    │ ├── cats
    │ └── dogs
    └── test
      ├── cats
      └── dogs

    '''
    image_list = glob.glob(str(pathlib.Path('images/raw/train/*')))
    move_path = 'images/cats_vs_dogs'
    move_categories = ['cat', 'dog']
    resize = (128,128)
    split = (.8, .1, .1)
    ih.move_and_split(image_list, move_path, move_categories, split, resize)
    
    # Check the split of the images
    ih.count_images('images/cats_vs_dogs')
    ##################
    
    ##################
    # Image Generators
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    batch_size = 32
    image_path_train = str(pathlib.Path('images/cats_vs_dogs/train'))
    image_path_val = str(pathlib.Path('images/cats_vs_dogs/val'))
    image_path_test = str(pathlib.Path('images/cats_vs_dogs/test'))
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       width_shift_range=[-10,10],
                                       height_shift_range=0.2,
                                       brightness_range=[0.8,1.2],
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       rotation_range=180)
    train_datagen = train_datagen.flow_from_directory(
                   	image_path_train,
                   	target_size=(image_width, image_height),
                    batch_size = batch_size,
                   	class_mode='categorical')
    
    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_datagen = val_datagen.flow_from_directory(
            	image_path_val,
            	target_size=(image_width, image_height),
                batch_size = batch_size,
            	class_mode='categorical')
    
    print(train_datagen.class_indices)
    print(val_datagen.class_indices)
    ##################
    
    ##################
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Make all layers untrainable
    for layer in base_inception.layers:
        layer.trainable = False
                
    #last_output = base_inception.output                 
    last_output = base_inception.get_layer('mixed7').output
    x = Flatten()(last_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x) 
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_inception.input, outputs=predictions)
    
    
    inception_layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    print(pd.DataFrame(inception_layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']))
    
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy']) 
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    history = model.fit(train_datagen, 
                    steps_per_epoch = len(train_datagen.labels)//batch_size,
                    epochs=20, 
                    validation_data = val_datagen, 
                    validation_steps = len(val_datagen.labels)//batch_size,
                    verbose = 1,
                    callbacks=[stop_training])
    ##################
        
    ##################
    epochs = 20
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))
    
    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))
    
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()
    ##################
