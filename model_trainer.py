import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
import glob
import segmentation_models_3D as sm
import tensorflow as tf
from tensorflow import keras

from data_generator import image_loader
from unet_3d import simple_unet_model

print("Starting Now....Imported Libraries")
train_image_directory = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_directory = "BraTS2020_TrainingData/input_data_128/train/masks/"
val_image_directory = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_directory  = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_image_list = os.listdir(train_image_directory)
train_mask_list = os.listdir(train_mask_directory)
val_image_list = os.listdir(val_image_directory)
val_mask_list = os.listdir(val_mask_directory)

batch_size = 16

train_image_data_generator = image_loader(train_image_directory,
                                          train_image_list,
                                          train_mask_directory,
                                          train_mask_list,
                                          batch_size)

val_image_data_generator = image_loader(val_image_directory,
                                        val_image_list,
                                        val_mask_directory,
                                        val_mask_list,
                                        batch_size)

metrics = ['accuracy']

learning_rate = 0.0001

optim = keras.optimizers.Adam(learning_rate=learning_rate)

steps_per_epoch = len(train_image_list)//batch_size
val_steps_per_epoch = len(val_image_list)//batch_size

model = simple_unet_model(128,128,128,3,4)

model.compile(optimizer = optim,loss=tf.keras.losses.CategoricalCrossentropy(),metrics=metrics)
print(model.summary)

print(model.input_shape)
print(model.output_shape)

history=model.fit(train_image_data_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          verbose=1,
          validation_data=val_image_data_generator,
          validation_steps=val_steps_per_epoch,
          )
model.save('brats_3d.hdf5')
print('Model Saved....Exiting...')