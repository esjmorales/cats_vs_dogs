# -*- coding: utf-8 -*-
"""
@author: EJMorales

File with a few functions for helping us handle the images for our model

move_and_split() arranges the images into directories for the DataGenerator
count_images() displays the counts of each category after the split
"""


import glob
import PIL
import pathlib
import shutil
import os
import random
import numpy as np


def copy_images(image_list, copy_path, resize=None):
    '''
    Parameters
    ----------
    image_list : List
        List of images, full path
    copy_path : string
        Folder location where to copy images to
        If folder does not exist, will be created
    resize : (width, height), optional
        Resize images to specified width and height

    Returns
    -------
    None.

    '''
    
    # Make the directory if it doesn't eixt
    destination = str(pathlib.Path(copy_path))
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(image_list):
        print(str(i+1), len(image_list), img, sep = " : ")
        save_path = os.path.join(destination, pathlib.Path(img).name)
        if resize is None:
            shutil.copyfile(img, save_path)
        else:
            img_temp = PIL.Image.open(img).resize(resize)
            img_temp.save(save_path)
            

def move_and_split(image_list, move_path, move_categories, split, resize=None):
    '''
    Parameters
    ----------
    image_list : List
        List of images, full path
    move_path : TYPE
        DESCRIPTION.
    move_categories : list
        names of the categories (sub folders) to create and move images to
        Also used to filter root images
    split : string
        Folder location where to copy images to
        If folder does not exist, will be created
        Sub folders train, val, test created
    resize : (width, height), optional
        Resize images to specified width and height

    Returns
    -------
    None.

    '''
    #######################
    # Check if the splits add up to 1, should be in percentages
    if sum(split) != 1:
        print("Split percentages exceed 100%")
        return 
    
    random.shuffle(image_list)
    #######################
    
    #######################
    # Split the masks
    # How many total images do we have? 
    # How many of each type?
    data_size = len(image_list)
    train_size = int(split[0] * data_size)
    val_size = int(split[1] * data_size)
    test_size = data_size - train_size - val_size
        
    train_images = image_list[:train_size]
    val_images = image_list[train_size:(train_size+val_size)]  
    test_images = image_list[(train_size+val_size):]
    #######################
    
    #######################
    # Make the directory if it doesn't exist
    destination = str(pathlib.Path(move_path))
    pathlib.Path(destination).mkdir(parents=True, exist_ok=True)
    
    # Create a train, val, test directory for each category
    # Move images into subfolders
    for f in ['train', 'val', 'test']:
        for c in move_categories:
            # We'll pass the lists to be copied
            # We assume the images have the category name in them
            # We use this assumption to filter
            images_to_copy = []
            if f == 'train':
                images_to_copy = train_images
            elif f == 'val':
                images_to_copy = val_images
            elif f == 'test':
                images_to_copy = test_images
            images_to_copy = [img for img in images_to_copy if c in img]
            copy_path = os.path.join(move_path, f, c)
            pathlib.Path(copy_path).mkdir(parents=True, exist_ok=True)
            
            copy_images(images_to_copy, copy_path, resize)
    #######################
    
    #######################
    print(f"{'Dataset size':>20}: {data_size:>4}")
    print(f"{'Training size':>20}: {train_size:>4}")
    print(f"{'Validation size':>20}: {val_size:>4}")
    print(f"{'Test size':>20}: {test_size:>4}")
    #######################
            
            
def get_image_dimensions(image_list):
    '''
    Parameters
    ----------
    image_list : List
        List of images, full path
        
    Returns
    -------
    Numpy array of with [width,height]
    '''
    image_sizes = np.zeros(shape=(len(image_list),2))
    
    for i, img in enumerate(image_list):
        print(str(i+1), len(image_list), img, sep = " : ")
        img_size = np.array(PIL.Image.open(img).size)
        image_sizes[i] = img_size
        
    return (image_sizes)

def count_images(directory):
    '''
    Parameters
    ----------
    directory : string
        Root directory of where the test/val/test folders are
        with respective sub-folder for each category
        
    Returns
    -------
    None
    '''        
    for folder in ('train', 'val', 'test'):
        path = os.path.join(directory, folder)
        total = len(list(pathlib.Path(path).rglob("*.jpg")))
        print(folder)
        print(f"{'Total':>15}: {total:>6}")
        list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
        for c in list_subfolders_with_paths:
            count = len(glob.glob(str(c + '/*.jpg')))
            print(f"{os.path.basename(os.path.normpath(c)):>15}: {count:>6} {round((count/total)*100,2):>6}%")    