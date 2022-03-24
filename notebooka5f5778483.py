import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
COPY_PATH = r"C:\Users\Михаил\Documents\ml-vechicles-and-plants-challenge"
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [90,90]
EPOCHS = 100

# # PATHS TO IMAGES
from os import makedirs
from os import listdir
PATH = r"C:\Users\Михаил\Documents\ml-vechicles-and-plants-challenge\data"
IMGS = listdir(PATH); 
print('There are %i images '%(len(IMGS)))

import json
from pandas.io.json import json_normalize #package for flattening json in pandas df

#load json object
with open(r'C:\Users\Михаил\Documents\ml-vechicles-and-plants-challenge\data.json') as f:
    d = json.load(f)

#lets put the data into a pandas df
data = pd.json_normalize(d, 'initial_bundle')
test = pd.json_normalize(d, 'test_bundle')

# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = COPY_PATH +'/Vehicles_vs_Plants/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['Vehicles/', 'Plants/','Other/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
# copy datasets images into subdirectories
src_directory = PATH
for file in listdir(src_directory):
    src = src_directory + file
    if 'data/' + file in test.file.tolist():
        dst_dir = 'test/'
        d = test
    else:
        dst_dir = 'train/'
        d = data
    if any(d[d.file == 'data/' + file]['category.name'] == 'Vehicles'):
        dst = dataset_home + dst_dir + 'Vehicles/' + file
        copyfile(src, dst)
    elif any(d[d.file == 'data/' + file]['category.name'] == 'Plants'):
        dst = dataset_home + dst_dir + 'Plants/' + file
        copyfile(src, dst)
    else:
        dst = dataset_home + dst_dir + 'Other/'  + file
        copyfile(src, dst)

filenames = tf.io.gfile.glob(str( + "/Vehicles_vs_Plants/train/*/*.jp*g"))
filenames.extend(tf.io.gfile.glob(str(COPY_PATH + "/Vehicles_vs_Plants/train/*/*.png")))
train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
test_filenames = tf.io.gfile.glob(str(COPY_PATH + "/Vehicles_vs_Plants/test/*/*.jp*g"))
test_filenames.extend(tf.io.gfile.glob(str(COPY_PATH + "/Vehicles_vs_Plants/test/*/*.png")))

COUNT_OTHER = len([filename for filename in train_filenames if "Other" in filename])
print("Other images count in training set: " + str(COUNT_OTHER))

COUNT_PLANTS = len([filename for filename in train_filenames if "/Plants" in filename])
print("Plants images count in training set: " + str(COUNT_PLANTS))

COUNT_VEHICLES = len([filename for filename in train_filenames if "/Vehicles/" in filename])
print("Vehicles images count in training set: " + str(COUNT_VEHICLES))

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

for f in train_list_ds.take(5):
    print(f.numpy())

TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))

CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str(COPY_PATH + "/Vehicles_vs_Plants/train/*"))])
CLASS_NAMES

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    if parts[-2] == "Other":
        return 0
    elif  parts[-2] == "Plants":
        return 1
    else:
        return 2

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
    img = tf.cond(
    tf.image.is_jpeg(img),
    lambda: tf.image.decode_jpeg(img, channels=3),
    lambda: tf.image.decode_png(img, channels=3))
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_list_ds = tf.data.Dataset.list_files(str(COPY_PATH + "/Vehicles_vs_Plants/test/*/*"))
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
TEST_IMAGE_COUNT

def prepare_for_training(ds, cache=True, shuffle_buffer_size=2000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(16):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n] == 0:
            plt.title("OTHER")
        elif label_batch[n] == 1:
            plt.title("PLANTS")
        else :
            plt.title("VEHICLES")
        plt.axis("off")

show_batch(image_batch.numpy(), label_batch.numpy())

initial_bias = np.log([COUNT_OTHER/COUNT_PLANTS])
initial_bias

weight_for_0 = (1 / COUNT_OTHER)*(TRAIN_IMG_COUNT)/3.0 
weight_for_1 = (1 / COUNT_PLANTS)*(TRAIN_IMG_COUNT)/3.0
weight_for_2 = (1 / COUNT_VEHICLES)*(TRAIN_IMG_COUNT)/3.0

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
print('Weight for class 2: {:.2f}'.format(weight_for_2))

from tensorflow import keras
from tensorflow.python.keras import layers
model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=4, kernel_size=3, activation='relu', padding='same',
                  input_shape=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

#     Block Three
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    #Dense layers
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(36, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(9, activation='relu'),
    layers.Dense(3, activation='softmax'),
])

model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics='sparse_categorical_accuracy'
    )

history = model.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // (BATCH_SIZE),
    epochs=100,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // (BATCH_SIZE),
    class_weight=class_weight,
)

import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()

pred = model.predict(test_ds)
predictions = np.argmax(pred, axis=-1)
probability = np.max(pred, axis=-1)
labels = {1:'Plants',2:'Vehicles'}
len([print(test_filenames[i].split('/')[-1], probability[i], labels[predictions[i]]) for i, v in enumerate(predictions.tolist()) if predictions[i] == 1 or predictions[i] == 2])