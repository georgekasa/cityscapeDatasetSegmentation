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
        14: ("polegroup", 153, 153, 153),
        15: ("traffic light", 250, 170, 30),
        16: ("traffic sign", 220, 220, 0),
        17: ("vegetation", 107, 142, 35),
        18: ("terrain", 152, 251, 152),
        19: ("sky", 70, 130, 180),
        20: ("person", 220, 20, 60),
        21: ("rider", 255, 0, 0),
        22: ("car", 0, 0, 142),
        23: ("truck", 0, 0, 70),
        24: ("bus", 0, 60, 100),
        25: ("caravan", 0, 0, 90),
        26: ("trailer", 0, 0, 110),
        27: ("train", 0, 80, 100),
        28: ("motorcycle", 0, 0, 230),
        29: ("bicycle", 119, 11, 32),
        30: ("license plate", 0, 0, 142)
    }
    df = pd.DataFrame.from_dict(id_map, orient='index', columns=["className",'r', 'g', 'b'])
    #df.index.name = 'name'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id_name'}, inplace=True)
    print(df)
    return id_map

#inferno, plasma, magma, viridis
def printImages(dataset, index_of_dataset , numberofImages):
  x = dataset.take(index_of_dataset)
  for image, labels, labels_truth in x:
      fig, axs = plt.subplots(numberofImages, 3, figsize=(16, 16))
      for i in range(numberofImages):
          # Convert the tensor to a NumPy array
          image_array = (255*image[i]).numpy().astype("uint8")
          labels_array = labels[i].numpy().astype("uint8")
          labels_truth_array = labels_truth[i].numpy().astype("uint8")
          axs[i, 0].imshow(image_array)
          axs[i, 0].axis("off")
          axs[i, 0].set_title("Image {}".format(i))
          axs[i, 1].imshow(labels_array, cmap="viridis")
          axs[i, 1].axis("off")
          axs[i, 1].set_title("mask generated {}".format(i))
          axs[i, 2].imshow(labels_truth_array, cmap="plasma")
          axs[i, 2].axis("off")
          axs[i, 2].set_title("mask truth {}".format(i))


      plt.show()
      break


def printImages2(dataset, index_of_dataset , numberofImages):
  x = dataset.take(index_of_dataset)
  for image, labels in x:
      fig, axs = plt.subplots(numberofImages, 3, figsize=(16, 16))
      for i in range(numberofImages):
          # Convert the tensor to a NumPy array
          image_array = (255*image[i]).numpy().astype("uint8")
          labels_array = labels[i].numpy().astype("uint8")
          axs[i, 0].imshow(image_array)
          axs[i, 0].axis("off")
          axs[i, 0].set_title("Image {}".format(i))
          axs[i, 1].imshow(labels_array, cmap="viridis")
          axs[i, 1].axis("off")
          axs[i, 1].set_title("mask generated {}".format(i))


      plt.show()
      break




def set_numeric_values(numeric_values, id_map):
  for _, info in id_map.items():
    # Extract the numeric values from the tuple and append them to the list
      numeric_values.extend(info[1:])

  # Convert the list to a NumPy array this can be done in one line
  numeric_values = np.array(numeric_values)
  


def get_numeric_array():
  return numeric_values

def preprocessEucledian(theImage, id_map):
    #x = time.time()
    #img = img_to_array(load_img(path, target_size=(256, 512)))#numpy returns
    image = tf.cast(theImage, dtype=tf.float32)
    data_image = image[:,:, :256, :] / 255.0
    data_mask = image[:,:, 256:, :]
    data_mask = tf.cast(data_mask, dtype=tf.float32)
    data_mask_truth = data_mask
    numeric_values = get_numeric_array()

    # Loop over the items in the id_map dictionary
    #for _, info in id_map.items():
    # Extract the numeric values from the tuple and append them to the list
    # numeric_values.extend(info[1:])

  # Convert the list to a NumPy array this can be done in one line
    #numeric_array = np.array(numeric_values)
    #numeric_array = np.reshape(numeric_array, (len(id_map), 3))


    class_rgb = tf.zeros((len(id_map), 3), dtype=tf.float32)#it may want 32,(31,3)
    class_rgb = numeric_values + class_rgb
    data_mask = tf.expand_dims(data_mask, axis=3)
    # Convert the mask to categorical format
    #mask = tf.zeros((*data_mask.shape[:2+1], num_classes), dtype=tf.int32)
    mask = tf.linalg.norm(data_mask - class_rgb, axis=-1)
    #for i in range(data_image.shape[0]): 

      #mask[i,...] = tf.linalg.norm(data_mask[i,:,:, None] - class_rgb, axis=-1)
    mask = tf.argmin(mask, axis = -1)

    #plt.imshow(cupy.asnumpy(mask[:,:]).astype("uint8"))
    #plt.show()
    #print(time.time() - x)
    return data_image, tf.cast(mask, tf.int32), data_mask_truth


#tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2, seed=seed)
def Augment(images, labels):
  print(images.shape)
  print(labels.shape)
  data_augmentation = tf.keras.layers.RandomFlip(mode="horizontal", seed=69)

  labels = tf.expand_dims(labels, axis=3)
  return data_augmentation(images), data_augmentation(labels)


def read_image_and_annotation(big_image, masks):
  '''
  Casts the image and annotation to their expected data type and
  normalizes the input image so that each pixel is in the range [-1, 1]

  Args:
    image (numpy array) -- input image
    annotation (numpy array) -- ground truth label map

  Returns:
    preprocessed image-annotation pair
  '''
  #print("hello")

  #big_image_cupy = cupy.asarray(big_image)
  #num_classes = len(id_map)
  image, annotation, annotation_truth = preprocessEucledian(big_image, id_map)
  #image = tf.convert_to_tensor(cupy.asnumpy(image), dtype=tf.float32)
  #annotation = tf.convert_to_tensor(cupy.asnumpy(annotation), dtype=tf.int32)


  return image, annotation#, annotation_truth #<- need for plot only!!!!!!!!!!!!!!!!!!!!!!!!!!!


from tensorflow.keras.utils import load_img, img_to_array
numeric_values = []
id_map = decalreIdmap()
#################################
###########load data#############
#################################
H = W = 256
batch_size = 4
seed_number = 123
pathData = "/media/gkasap/ssd256gb/datasets/cityscapes_data"
train_ds = tf.keras.utils.image_dataset_from_directory(
  pathData+"/trainDir",
  validation_split=0.1,
  subset="training",
  seed=seed_number,
  image_size=(H, W*2),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  pathData+"/valDir",
  validation_split=0.8,
  subset="validation",
  seed=seed_number,
  image_size=(H, W*2),
  batch_size=batch_size)



set_numeric_values(numeric_values, id_map)
numeric_values = np.reshape(numeric_values, (len(id_map), 3))
numeric_values = tf.convert_to_tensor(numeric_values, dtype=tf.float32)
#training_dataset = train_ds.map(read_image_and_annotation)
#validation_dataset = val_ds.map(read_image_and_annotation)
#x = training_dataset.map(Augment)
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
#printImages2(x, 10 , 4)
#training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
training_dataset = (
    train_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .repeat()
    .map(read_image_and_annotation)
    .prefetch(buffer_size=tf.data.AUTOTUNE))

validation_dataset = (
    val_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .repeat()
    .map(read_image_and_annotation)
    .prefetch(buffer_size=tf.data.AUTOTUNE))


# With this option, your preprocessing will happen on CPU, asynchronously, and will be buffered before going into the model. 
# In addition, if you call dataset.prefetch(tf.data.AUTOTUNE) on your dataset, the preprocessing will happen efficiently in parallel with training:
#https://www.tensorflow.org/guide/keras/preprocessing_layers






######################
#######create NN######
######################
#printImages()
channels = 3
num_classes = len(id_map)
# y_train = keras.utils.to_categorical(Y_train, num_classes)
# y_valid = keras.utils.to_categorical(Y_valid, num_classes)
W = H = 256
input_size = (H, W, channels)
def conv2d_block(input_tensor, n_filters, kernel_size=3):
  """
  Adds convolutional layers with the parameters passed to it.

  Args:
    input_tensor (tensor) -- the input tenor
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size of the convolution
  """
  # first layer 
  x = input_tensor
  for i in range(2):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal", padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)

  return x
def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
  """
  Adds two convolutional blocks and then perform sampling on output of convolution.

  Args:
    input_tensor (tensor) -- the input tensor
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size of convolution
  
  Returns:
    f - the output features of the convolution block
    p - the maxpooled features with dropout
  """

  f = conv2d_block(inputs, n_filters=n_filters)
  p = tf.keras.layers.MaxPool2D(pool_size=(2,2))(f)
  p = tf.keras.layers.Dropout(0.3)(p)

  return f, p

def encoder(inputs):
  """
  This function defines the encoder or downsampling path.

  Args:
    inputs (tensor) -- batch of input images

  Returns:
    p4 - the output maxpooled features of the last encoder block
    (f1, f2, f3, f4) - the output features of all the encoder blocks
  """
  f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3)
  f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2, 2), dropout=0.3)
  f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2, 2), dropout=0.3)
  f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2, 2), dropout=0.3)

  return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
  """
  This function defines the bottleneck convolutions to extract more features before the unsampling layers.
  """

  bottle_neck = conv2d_block(inputs, n_filters=1024)

  return bottle_neck

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
  """
  Defines the one decoder block of the UNet

  Args:
    inputs (tensor) -- batch of input features
    conv_output (tensor) -- features from an encoder block
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size
    strides (int) -- strides for the deconvolution/upsampling
    padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

  Returns:
    c (tensor) -- output features of the decoder block
  """
  u = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size,
                                      strides=strides, padding="same")(inputs)
  c = tf.keras.layers.concatenate([u, conv_output])                                      
  c = tf.keras.layers.Dropout(dropout)(c)
  c = conv2d_block(c, n_filters=n_filters, kernel_size=3)

  return c

def decoder(inputs, convs, output_channels):
  """
  Defines the decoder of the UNet chaining together 4 decoder blocks. 
  
  Args:
    inputs (tensor) -- batch of input features
    convs (tuple) -- features from the encoder blocks
    output_channels (int) -- number of classes in the label map

  Returns:
    outputs (tensor) -- the pixel wise label map of the image
  """

  f1, f2, f3, f4 = convs
  # 5 is the bottleneck if you ask 
  c6 = decoder_block(inputs, conv_output=f4, n_filters=512, kernel_size=(3,3),
                     strides=(2,2), dropout=0.3)
  
  c7 = decoder_block(c6, conv_output=f3, n_filters=256, kernel_size=(3,3),
                     strides=(2,2), dropout=0.3)
  
  c8 = decoder_block(c7, conv_output=f2, n_filters=128, kernel_size=(3,3),
                     strides=(2,2), dropout=0.3)
  
  c9 = decoder_block(c8, conv_output=f1, n_filters=64, kernel_size=(3,3),
                     strides=(2,2), dropout=0.3)
  
  outputs = tf.keras.layers.Conv2D(output_channels, (1,1), activation="softmax",
                                   bias_initializer = keras.initializers.Constant(-np.log(1/output_channels)))(c9)

  return outputs


OUTPUT_CHANNELS = 31

def unet():
  """
  Defines the UNet by connecting the encoder, bottleneck and decoder.
  """

  # specify the input shape
  inputs = tf.keras.layers.Input(shape=(H, W, 3))

  # feed the inputs to the encoder
  encoder_output, convs = encoder(inputs)

  # feed the encoder output to the bottleneck
  bottle_neck = bottleneck(encoder_output)

  # feed the bottleneck and encoder block outputs to the decoder
  # specify the number of classes ia the `output_channels` argument
  outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)

  # create the model
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model

# instantiate the model
model = unet()
# print(model.summary())


NameoftheSimulation = "my_model_noAug_AccuracyDiceNvidia_v4"
############################################################################################################
###################### set up call backs####################################################################
############################################################################################################
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=30,
    mode='max',
    restore_best_weights=True)

from datetime import datetime
csv_logger = tf.keras.callbacks.CSVLogger("/home/gkasap/Documents/Python/projects/DLfullProject/logs"+ "training"+NameoftheSimulation+".log")

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + NameoftheSimulation

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/gkasap/Documents/Python/projects/DLfullProject/' + NameoftheSimulation +'h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
    )
# Saves the current weights after every epoch
CALLBACKS = [early_stopping, csv_logger, tensorboard_callback, model_checkpoint]
############################################################################################################
########################## metrics ########################################################################
from tensorflow.keras import backend as K
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
#2 kai 13, 0.9838

#expand_dims
#IoU = TP / (TP + FP + FN)
def mean_iou(y_true, y_pred):
    num_classes = K.int_shape(y_pred)[-1]


    indices = K.argmax(y_pred, -1)
    indices = K.reshape(indices, [-1, 256, 256, 1])
    true_cast = K.expand_dims(y_true, axis=-1)
    y_true_class = true_cast[..., 1]



    indices_cast = K.cast(indices, dtype='float32')
    true_cast = K.expand_dims(y_true, axis=-1)
    true_cast = K.cast(true_cast, dtype='float32')


    iou_list = []
    axis = [1, 2, 3]
    intersection = K.sum(true_cast * indices_cast, axis=axis)


    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        y_pred_class = K.argmax(y_pred_class, axis=-1)
        intersection = K.sum(y_true_class * y_pred_class)
        union = K.sum(y_true_class) + K.sum(y_pred_class) - intersection
        iou = intersection / (union + K.epsilon())
        iou_list.append(iou)
    mean_iou = K.mean(K.stack(iou_list))
    return mean_iou



EPOCHS = 10
model.compile(optimizer=Adam(lr=1e-3), loss='sparse_categorical_crossentropy',
               metrics=[dice_coef_NVIDIA_multiClass, 'accuracy'])
# #print(model.summary())
# BATCH_SIZE = 8
# print(y_pred_f.dtype, y_true_f.dtype)
# 0.8
steps_per_epoch = 2678//batch_size
validation_steps = 400//batch_size

history = model.fit(
    training_dataset, 
    epochs=EPOCHS, 
    steps_per_epoch=steps_per_epoch,
    verbose=1, shuffle=True,
    callbacks = CALLBACKS,
    validation_data=validation_dataset,
    validation_steps=validation_steps
)
model.save(NameoftheSimulation+'.h5')
