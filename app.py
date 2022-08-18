import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

database_dir = r'E:/Database/Image_Super_Resolution/Raw Data'
SIZE = (256, 256)
batch_size = 1

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key = alphanum_key)

def load_data(database_dir):
    high_res_path = os.path.join(database_dir, 'high_res')
    low_res_path = os.path.join(database_dir, 'low_res')
    high_img_path = []
    files = os.listdir(high_res_path)
    files = sorted_alphanumeric(files)
    for i in tqdm(files):
        high_img_path.append(os.path.join(high_res_path, i))
    low_img_path = []
    files = os.listdir(low_res_path)
    files = sorted_alphanumeric(files)
    for i in tqdm(files):
        low_img_path.append(os.path.join(low_res_path, i))
    return high_img_path, low_img_path

high_img_path, low_img_path = load_data(database_dir)

hSeries = pd.Series(high_img_path, name = 'predicted')
lSeries = pd.Series(low_img_path, name = 'original')

df = pd.concat([lSeries, hSeries], axis = 1)

train_df, dummy_df = train_test_split(df, train_size = .75, shuffle = True, random_state = 123)
valid_df, test_df = train_test_split(dummy_df, train_size = .5, shuffle = True, random_state = 123)

t_t_and_v_gen = ImageDataGenerator(rescale = 1. / 255)

train_gen = t_t_and_v_gen.flow_from_dataframe(train_df, x_col = 'original', y_col = 'predicted', class_mode = 'input', target_size = SIZE, color_mode = 'rgb', shuffle = True, batch_size = batch_size)
valid_gen = t_t_and_v_gen.flow_from_dataframe(valid_df, x_col = 'original', y_col = 'predicted', class_mode = 'input', target_size = SIZE, color_mode = 'rgb', shuffle = True, batch_size = batch_size)
test_gen = t_t_and_v_gen.flow_from_dataframe(test_df, x_col = 'original', y_col = 'predicted', class_mode = 'input', target_size = SIZE, color_mode = 'rgb', shuffle = True, batch_size = batch_size)

def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample

def model():
    inputs = layers.Input(shape= [256, 256, 3])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    
    d5 = down(512,(3,3),True)(d4)
    #upsampling
    u1 = up(512,(3,3),False)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)

model = model()
model.summary()

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mean_absolute_error', metrics = ['accuracy'])

model.fit(x = train_gen, validation_data = valid_gen, epochs = 5, verbose = 1, shuffle = False)

def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)
    plt.show()

for i in range(5):
    test_image = next(test_gen)
    low_test_image = np.array(test_image[0]).reshape(256, 256,3)
    high_test_image = np.array(test_image[1]).reshape(256, 256,3)
    predicted = np.clip(model.predict(low_test_image.reshape(1, 256, 256, 3)), 0.0, 1.0).reshape(256, 256,3)
    plot_images(high_test_image, low_test_image, predicted)

