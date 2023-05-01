import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm
import warnings
import time
import time

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import cv2 as cv

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



def dice_coef_NVIDIA_multiClass(y_true, y_pred, num_classes =29, smooth=1.0):
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
    #union = K.cast(K.sum(y_true + y_pred, axis = axis), "float32")
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    dice = K.mean((2. * intersection + smooth) / (union + smooth))#axis = 0)
    return dice
#2 kai 13, 0.9838


#IoU = TP / (TP + FP + FN)
def mean_iou(y_true, y_pred, num_classes = 29, smooth=1.0):
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
    #union = K.cast(K.sum(y_true + y_pred, axis = axis), "float32")
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    mean_iou = K.mean((intersection + smooth) / (union - intersection + smooth))#axis = 0)

    return mean_iou


# Load the model
custom_objects = {'dice_coef_NVIDIA_multiClass': dice_coef_NVIDIA_multiClass, 'mean_iou': mean_iou}
model = load_model('my_model_noAug_AccuracyDiceNvidia_v5d.h5', custom_objects=custom_objects)

# Define paths
SAMPLE_VIDEO = '/home/gkasap/Documents/Python/projects/DLfullProject/mit_driveseg_sample.mp4'

# Read video frames
video = cv.VideoCapture(SAMPLE_VIDEO)
num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv.CAP_PROP_FPS))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
width = 256
height = 256
out = cv.VideoWriter('output.mp4', fourcc, fps, (width, height))

width = 256
height = 256

while True:
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the frame for the model
    input_image = cv.resize(frame, (256, 256)).astype(np.float32) / 255.0
    input_image_expanded = np.expand_dims(input_image, axis=0)

    # Make the segmentation prediction using the model
    prediction = model.predict(input_image_expanded)
    prediction = np.argmax(prediction, axis=-1)
    prediction = np.expand_dims(prediction, axis=-1)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.uint8(prediction)

    # Convert the class labels to a color map
    prediction_color = cv.applyColorMap(prediction, cv.COLORMAP_JET)
    input_image_uint8 = (input_image * 255).astype(np.uint8)
    # Combine the original frame and the mask
    combined_image = cv.addWeighted(input_image_uint8, 0.7, prediction_color, 0.3, 0)

    # Display the resulting frame
    cv.imshow('frame', combined_image)
    out.write(combined_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()


video.release()
out.release()
cv.destroyAllWindows()








