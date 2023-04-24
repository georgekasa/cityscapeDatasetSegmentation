# cityscapeDatasetSegmentation [1]

An image segmentation dataset with 31 classes [2]
the first version was a naive prediction with naive I mean
1) no image augmentation
2) without early stop, 10 epochs only and a smaller dataset due memory issues
3) the data manipulation of the picture e.g. the normaliz and the labeling correctness
there are 31 classes and are writen in this format:
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    etc.
in the masks they are many many pixels which their values could be e.g. (0,0,2) so I used the Euclidean distance from the cupy library to find the smaller distance to match this pixel to the correct label (e.g. here in an unlabeled), this task was extremely computationally, because the dataset has 3k masks with dim 256x256x3 so = 589_824_000 pixels I did the Euclidean distance with NVIDIA cupy which speed up the process comparing to the numpy by /10!!(for each picture manipulation, will try also to use cuda -python, the v1 is just a naive)
4) Also I used the Unet, algorithm without using some tricks which Karpathy has suggested from CS231N, the code for the NN I did get it from the course which I finished in coursera AdvancedComputer vision with tf & keras
    https://github.com/georgekasa/courseraAdvancedComputerVisionTensorFlow- 
    
 
######################### v2 #########################


1 is on the way
updated 2 with tf.keras.utils.image_dataset_from_directory
updated 3 used tensors 
 
 
 
 
[1]https://github.com/mcordts/cityscapesScripts
[2]https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

(i dont why the format is broken in edit is okay)
