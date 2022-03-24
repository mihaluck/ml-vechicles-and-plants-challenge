import pandas as pd
import json
from pandas.io.json import json_normalize #package for flattening json in pandas df

#load json object
with open('../input/autos-and-plants/data.json') as f:
    d = json.load(f)

#lets put the data into a pandas df
#clicking on raw_nyc_phil.json under "Input Files"
#tells us parent node is 'programs'
data = pd.json_normalize(d, 'initial_bundle')
test = pd.json_normalize(d, 'test_bundle')
# data[data['category.name'] == 'Vehicles']
# data[data['category.name'] == 'Plants']

# # PATHS TO IMAGES
from os import makedirs
from os import listdir
PATH = '../input/autos-and-plants/data/data/'
IMGS = listdir(PATH); 
print('There are %i images '%(len(IMGS)))

# load dogs vs cats dataset, reshape and save to a new file
# from os import listdir
# from numpy import asarray
# from numpy import save
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# # define location of dataset
# folder = 'data/data/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
#     # determine class
#     if data.file == file:
#         if ['category.name'] == 'Vehicles':
#             output = 1.0
#         elif ['category.name'] == 'Plants'
#             output = 0.0
#         labels.append(output)
#         # load image
# photo = load_img(folder + file, target_size=(200, 200))
# # convert to numpy array
# photo = img_to_array(photo)
# # store
# photos.append(photo)
        
# convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('dogs_vs_cats_photos.npy', photos)
# save('dogs_vs_cats_labels.npy', labels)

# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'dataset_Vechicles_vs_Plants/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
# create label subdirectories
    labeldirs = ['Vechicles/', 'Plants/','other/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
# copy datasets images into subdirectories
src_directory = '../input/autos-and-plants/data/data/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if data[data.file == 'data/' + file]['category.name'].any() == 'Vechicles':
        dst = dataset_home + dst_dir + 'Vechicls/' + file
    elif data[data.file == 'data/' + file]['category.name'].any() == 'Plants':
        dst = dataset_home + dst_dir + 'Plants/' + file
    else:
        dst = dataset_home + dst_dir + 'other'  + file
    copyfile(src, dst)

'data/'+file in data.file.tolist()

# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    dataset_home+'train/',
    image_size=[128, 128],
    labels =  "inferred",
    interpolation='nearest',
    batch_size=64,
    seed = 1,
    shuffle=True,
    validation_split = 0.2,
    subset='validation'
)
# ds_valid_ = image_dataset_from_directory(
#     '../input/autos-and-plants',
#     image_size=[128, 128],
#     interpolation='nearest',
#     batch_size=64,
#     shuffle=False,
# )

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# ds_valid = (
#     ds_valid_
#     .map(convert_to_float)
#     .cache()
#     .prefetch(buffer_size=AUTOTUNE)
# )

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    # YOUR CODE HERE
    layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding='same'),
    layers.MaxPool2D(),
    # ____,

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    # YOUR CODE HERE: Add loss and metric
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()