#https://www.kaggle.com/code/tr1gg3rtrash/car-driving-segmentation-unet-from-scratch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm
import warnings
import time
import time
import cupy
import pickle
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import backend as K


def plotResults(rangeStart, rangeEnd, Y_pred):
    fig, axes = plt.subplots(nrows=rangeEnd-rangeStart, ncols=3, figsize=(16, 16))
    # loop over the first 5 validation examples
    for i in range(rangeEnd - rangeStart):
        # plot the input image in the first column
        axes[i, 0].imshow((255.0*X_valid[rangeStart + i, ...]).astype("uint8"))
        axes[i, 0].set_title('Input Image')

        # plot the predicted mask in the second column
        predicted_mask = np.argmax(Y_pred[i], axis=-1)

        axes[i, 1].imshow(predicted_mask, cmap='viridis')
        axes[i, 1].set_title('Predicted Mask')

        # plot the correct mask in the second column

        axes[i, 2].imshow(Y_valid[rangeStart+i], cmap='viridis')
        axes[i, 2].set_title('True Mask')

        # turn off the axis labels
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')
    # adjust subplot spacing and display
    fig.tight_layout()
    #plt.axes("off")
    plt.show()

def plotResultsPicture(pictures, Y_pred):
    fig, axes = plt.subplots(len(pictures), ncols=2, figsize=(16, 16))
    # loop over the first 5 validation examples
    for i in range(0, len(pictures)):
        # plot the input image in the first column
        axes[i, 0].imshow((255*pictures[i]).astype("uint8"))
        axes[i, 0].set_title('Input Image')

        # plot the predicted mask in the second column
        predicted_mask = np.argmax(Y_pred[i], axis=-1)

        axes[i, 1].imshow(predicted_mask.astype("uint8"), cmap='viridis')
        axes[i, 1].set_title('Predicted Mask')



        # turn off the axis labels
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    # adjust subplot spacing and display
    fig.tight_layout()
    plt.show()



def decalreIdmap():
    id_map = {
        0: ("unlabelled", 0, 0, 0),
        1: ("static", 111, 74, 0),
        2: ("ground", 81, 0, 81),
        3: ("road", 128, 63, 127),
        4: ("sidewalk", 244, 35, 232),
        5: ("parking", 250, 170, 160),
        6: ("rail track", 230, 150, 140),
        7: ("building", 70, 70, 70),
        8: ("wall", 102, 102, 156),
        9: ("fence", 190, 153, 153),
        10: ("guard rail", 180, 165, 180),
        11: ("bridge", 150, 100, 100),
        12: ("tunnel", 150, 120, 90),
        13: ("pole", 153, 153, 153),
        14: ("traffic light", 250, 170, 30),
        15: ("traffic sign", 220, 220, 0),
        16: ("vegetation", 107, 142, 35),
        17: ("terrain", 152, 251, 152),
        18: ("sky", 70, 130, 180),
        19: ("person", 220, 20, 60),
        20: ("rider", 255, 0, 0),
        21: ("car", 0, 0, 142),
        22: ("truck", 0, 0, 70),
        23: ("bus", 0, 60, 100),
        24: ("caravan", 0, 0, 90),
        25: ("trailer", 0, 0, 110),
        26: ("train", 0, 80, 100),
        27: ("motorcycle", 0, 0, 230),
        28: ("bicycle", 119, 11, 32),
       # 30: ("license plate", 0, 0, 142)
    }
    df = pd.DataFrame.from_dict(id_map, orient='index', columns=["className",'r', 'g', 'b'])
    #df.index.name = 'name'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id_name'}, inplace=True)
    print(df)
    return id_map
#NVIDIA
def dice_coef_NVIDIA(y_true, y_pred, smooth=1):
    indices = K.argmax(y_pred, -1)
    indices = K.reshape(indices, [-1, 256, 256, 1])

    indices_cast = K.cast(indices, dtype='float32')
    true_cast = K.expand_dims(y_true, axis=-1)
    true_cast = K.cast(true_cast, dtype='float32')
    

    axis = [1, 2, 3]
    intersection = K.sum(true_cast * indices_cast, axis=axis)
    union = K.sum(true_cast, axis=axis) + K.sum(indices_cast, axis=axis)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice

def dice_coef(y_true, y_pred):
  smooth=1.0
  y_pred = (K.argmax(y_pred, -1))
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  y_true_f = K.cast(y_true_f, dtype='float32')
  y_pred_f = K.cast(y_pred_f, dtype='float32')
  intersection = K.sum(y_true_f * y_pred_f)
  dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  return dice

#IoU = TP / (TP + FP + FN)
def mean_iou(y_true, y_pred, num_classes = 29, smooth=1.0):
    #print(y_true.shape)
    #print(y_pred.shape)#None 256x256x31
    y_true = K.cast(y_true, dtype='int32') #None 256x256
    y_pred = K.argmax(y_pred, axis=-1)#None 256x256
    y_pred = K.expand_dims(y_pred, axis=-1) 
    y_true = K.cast(K.one_hot(y_true, num_classes), "int32")#None 256x256x31
    y_pred = K.cast(K.one_hot(y_pred, num_classes), "int32")#None 256x256x31
    y_true = tf.squeeze(y_true, axis=-2)
    y_pred = tf.squeeze(y_pred, axis=-2)
    #y_pred = K.flatten(y_pred)
    axis = [1, 2]
    intersection = K.cast(K.sum(y_true * y_pred, axis = axis), "float32")
    #print(intersection.shape)
    print(intersection)
    #union = K.cast(K.sum(y_true + y_pred, axis = axis), "float32")
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    #print(type(smooth))
    mean_iou = K.mean((intersection + smooth) / (union - intersection + smooth))#axis = 0)

    return mean_iou

def dice_coef_NVIDIA_multiClass(y_true, y_pred, num_classes = 31, smooth=1.0):
    #print(y_true.shape)
    #print(y_pred.shape)#None 256x256x31
    y_true = K.cast(y_true, dtype='int32') #None 256x256
    y_pred = K.argmax(y_pred, axis=-1)#None 256x256
    y_true = K.cast(K.one_hot(y_true, num_classes), "int32")#None 256x256x31
    y_pred = K.cast(K.one_hot(y_pred, num_classes), "int32")#None 256x256x31

    #y_pred = K.flatten(y_pred)
    axis = [1, 2]
    intersection = K.cast(K.sum(y_true * y_pred, axis = axis), "float32")
    #print(intersection.shape)
    print(intersection)
    #union = K.cast(K.sum(y_true + y_pred, axis = axis), "float32")
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    #print(type(smooth))
    dice = K.mean((2. * intersection + smooth) / (union + smooth))#axis = 0)
    print(union)
    return dice

the_path = r"/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/my_data.pkl"



with open(the_path, "rb") as f:
    loaded_data = pickle.load(f)

X_train =  np.array(loaded_data["x_train"], dtype="float32")
Y_train = np.array(loaded_data["y_train"], dtype="int32")
X_valid = np.array(loaded_data["x_valid"], dtype="float32")
Y_valid = np.array(loaded_data["y_valid"], dtype="int32")
##plt.imshow(Y_train[0].astype("uint8"))


# Calculate the crop indices

# start_row = (X_train.shape[1] - 128) // 2
# start_col = (X_train.shape[2] - 128) // 2
# end_row = start_row + 128
# end_col = start_col + 128

# # Crop the input array
# X_train = X_train[:,start_row:end_row, start_col:end_col,3]
# X_train = X_train[:700, :, :, :]
# Y_train = Y_train[:700, :, :]
# X_valid = X_valid[:200, :, :, :]
# Y_valid = Y_valid[:200, :, :]

#/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes
#car.jpg  drone_1.jpg  my_data.pkl  video.mp4
# load the saved model
custom_objects = {'dice_coef_NVIDIA_multiClass': dice_coef_NVIDIA_multiClass, 'mean_iou': mean_iou}
model = load_model('my_model_noAug_AccuracyDiceNvidia_v5d.h5', custom_objects=custom_objects)

import cv2
# Load image
img1 = cv2.imread("/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/drone_1.jpg")
img2 = cv2.imread("/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/car.jpg")
img3 = cv2.imread("/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/car_old.jpeg")
img4 = cv2.imread("/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/many_cars.png")
# Resize image
resized_img = np.zeros((4,256, 256, 3))

resized_img[0] = cv2.cvtColor(cv2.resize(img1, (256, 256)).astype(np.float32)/255.0, cv2.COLOR_BGR2RGB)
resized_img[1] = cv2.cvtColor(cv2.resize(img2, (256, 256)).astype(np.float32)/255.0, cv2.COLOR_BGR2RGB)
resized_img[2] = cv2.cvtColor(cv2.resize(img3, (256, 256)).astype(np.float32)/255.0, cv2.COLOR_BGR2RGB)
resized_img[3] = cv2.cvtColor(cv2.resize(img4, (256, 256)).astype(np.float32)/255.0, cv2.COLOR_BGR2RGB)
new_arr = resized_img[np.newaxis, ...]


# rangeStart = 300
# rangeEnd = 305
rangeStart = 89
rangeEnd = rangeStart + 5
Y_pred = model.predict(resized_img)
plotResultsPicture(resized_img, Y_pred)
#Y_pred = model.predict(X_valid[rangeStart:rangeEnd])
#plotResults(rangeStart, rangeEnd, Y_pred)

