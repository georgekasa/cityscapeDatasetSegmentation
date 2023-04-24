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


#num_classes = len(id_map.keys())
#https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py labels!!!

from tensorflow.keras.utils import load_img, img_to_array



import cupy #/10 the time with cupy 
def preprocessEucledian(path, id_map):
    #x = time.time()
    img = img_to_array(load_img(path, target_size=(256, 512)))
    data_image = img[:, :256, :] / 255.0
    data_mask = img[:, 256:, :]
    data_mask = cupy.asarray(data_mask, dtype=cupy.int32)
    class_rgb = cupy.zeros((len(id_map), 3), dtype=cupy.int32)
    for i, info in id_map.items():
        class_rgb[i] = cupy.array([info[1], info[2], info[3]])

    num_classes = len(id_map)
    # Convert the mask to categorical format
    mask = cupy.zeros((*data_mask.shape[:2], num_classes), dtype=cupy.uint8)

    mask = cupy.linalg.norm(data_mask[:,:, None] - class_rgb, axis=-1)
    mask = cupy.argmin(mask, axis = -1)

    #plt.imshow(cupy.asnumpy(mask[:,:]).astype("uint8"))
    #plt.show()
    #print(time.time() - x)
    return data_image, mask


def prepare_tensor_dataset(train_path, val_path, id_map):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    #class_rgb = cupy.asarray(class_rgb)

    for file in tqdm(os.listdir(train_path)):
        img, mask = preprocessEucledian(f"{train_path}/{file}", id_map)
        X_train.append(img)
        Y_train.append(mask)
    
    for file in tqdm(os.listdir(val_path)):
        img, mask = preprocessEucledian(f"{val_path}/{file}", id_map)
        X_val.append(img)
        Y_val.append(mask)

    return X_train, Y_train, X_val, Y_val


from tensorflow.keras.utils import to_categorical
id_map = decalreIdmap()
# start = time.time()
# X_train, Y_train, X_valid, Y_valid = prepare_tensor_dataset("/media/gkasap/ssd256gb/datasets/cityscapes_data/train",
#                                                              "/media/gkasap/ssd256gb/datasets/cityscapes_data/val", id_map)
# print(f"Time taken: {time.time() - start}")

# cupy_arrays = [Y_train, Y_valid]
# numpy_arrays = []

# for cupy_array in cupy_arrays:
#     numpy_everythink = []
#     for cupy_everything in cupy_array:
#         numpy_everythink.append(cupy_everything.get())
#     numpy_arrays.append(numpy_everythink)


# Store the arrays in a dictionary
# data = {"x_train": X_train, "y_train": numpy_arrays[0], "x_valid": X_valid, "y_valid": numpy_arrays[1]}
the_path = r"/home/gkasap/Documents/Python/projects/DLfullProject/cityscapes/my_data.pkl"
# Save the dictionary as a .pkl file
# with open(the_path, "wb") as f:
#     pickle.dump(data, f)

# Load the dictionary from the .pkl file
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
X_train = X_train[:700, :, :, :]
Y_train = Y_train[:700, :, :]
X_valid = X_valid[:200, :, :, :]
Y_valid = Y_valid[:200, :, :]

print("X train shape is:", X_train.shape)
print("Y train shape is:", Y_train.shape)
print("X valid shape is:", X_valid.shape)
print("Y valid shape is:", Y_valid.shape)
######################
#######create NN######
######################

channels = 3
num_classes = len(id_map)
# y_train = keras.utils.to_categorical(Y_train, num_classes)
# y_valid = keras.utils.to_categorical(Y_valid, num_classes)

input_size = (X_train.shape[1], X_train.shape[2], channels)
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
  
  outputs = tf.keras.layers.Conv2D(output_channels, (1,1), activation="softmax")(c9)

  return outputs


OUTPUT_CHANNELS = 31

def unet():
  """
  Defines the UNet by connecting the encoder, bottleneck and decoder.
  """

  # specify the input shape
  inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], 3))

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
print(model.summary())


EPOCHS = 10
model.compile(optimizer=Adam(lr=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#print(model.summary())
BATCH_SIZE = 8



history = model.fit(
    X_train, Y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1, 
    shuffle=True,
    validation_data=(X_valid, Y_valid)
)
model.save('my_model.h5')

#Y_pred = model.predict(X_valid)

# print some example predictions


#print("correct shape from linux 500*256*256*3 = 98304000 from python:", len(Y_valid)*len(Y_valid[0])*len(Y_valid[0][0])*len(Y_valid[0][0][0]))
#Y_train_one_hot = to_categorical(Y_train, num_classes)
#Y_valid_one_hot = to_categorical(Y_valid, num_classes)
#mask[..., np.argmin(np.abs(data_mask[0,0, None]-class_rgb), axis=-1)]
#mask[..., np.argmin(np.amin(class_diff[..., None], axis=-1))] = 1
#plt.imshow((255*mask[:,:,22]).astype("uint8"))

#np.linalg.norm(data_mask[0,0, None] - class_rgb, axis=-1) auto einai sosto
#np.linalg.norm(data_mask[:,:, None] - class_rgb, axis=-1).shape
#np.argmin(np.amin(class_diff[0,0], axis = -1))
#unique_values = np.amin(class_diff, axis = -1)
#mask[...,np.amin(unique_values, axis = -1)] = 1
#unique_values[np.argmax(value_counts)]



