import tensorflow as tf
import cv2
import PIL.Image as Image
import tensorflow.keras.preprocessing.image as img
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import copy

"""
cv2 reads and show in BGR (blue,green,red)
plt reads and show in RGB
PIL is also RGB
"""

"""
read_cv2: 

"""

im_cv= cv2.imread('images\\city.jpg')

"""
# normalize image

im_cv = im_cv/255


"""

# chanel is BGR which means cv2 returns image in bgr and we have to change the image rgb if we want
height, weidth, channel = im_cv.shape

# change channels to RGB
cv_rgb = cv2.cvtColor(im_cv,cv2.COLOR_BGR2RGB)

# if you show the image with cv2.imshow then again it returns the image to RGB even if you change
# the color map

## lets check to see if it works

# set the channel 1 and 2 to zero and open with plt.imshow or PIL to check if the first channel is red

im_cv[:,:,1]=0
im_cv[:,:,2]=0

plt.imshow(im_cv)

### As you see the image is red which means we were succesful to change the image to RGB






