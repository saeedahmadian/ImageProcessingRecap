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
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def makeModel():
    inp_layer = tf.keras.layers.Input(shape=(32,32,3))
    conv1_layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),activation='relu',
                                         strides=(1,1),padding='same',name='conv1_layer')
    conv1_out = conv1_layer(inp_layer)
    down_samp1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),name='max_pooling_1')
    down_samp_out= down_samp1(conv1_out)
    conv2_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',
                                         strides=(1,1),padding='same', name='conv2_layer')
    conv2_out = conv2_layer(down_samp_out)
    down_samp2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name='max_pooling_2')(conv2_out)
    conv3_layer= tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',
                                         strides=(1,1), padding='same', name='conv3_layer')
    conv3_out = conv3_layer(down_samp2)
    flat_layer= tf.keras.layers.Flatten(name='flatt_layer')(conv3_out)
    dens_layer= tf.keras.layers.Dense(units=64,activation='relu',name='dense_1')(flat_layer)
    output = tf.keras.layers.Dense(10,name='output')(dens_layer)

    return tf.keras.Model(inputs=inp_layer,outputs=output)


class Pred(tf.keras.Model):
    def __init__(self):
        super(Pred,self).__init__()
        self.layer1=




convmodel= makeModel()
convmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(name='CrossEntropy'),
                           'accuracy']
                  )

# y_labels= tf.keras.utils.to_categorical(train_labels)
results=convmodel.fit(x=train_images,y=train_labels,batch_size=64,epochs=1,
                      validation_data=(test_images,test_labels),
                      callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs'),
                                 tf.keras.callbacks.ModelCheckpoint('checkpoints')])



a=1
