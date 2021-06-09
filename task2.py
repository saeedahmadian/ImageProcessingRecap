import tensorflow as tf
import PIL
import pathlib
import glob, os
import matplotlib.pyplot as plt
import numpy as np

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(dataset_url.split('/')[-1].split('.')[0],dataset_url,untar=True)
data_dir= pathlib.Path(data_dir)

batch_size= 32
img_h = 180
img_w = 180

def SamplePlot(dataset):
    plt.figure(figsize=(10,10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplots(3,3,i+1)
            plt.imshow(images.numpy().astype(np.uint8))
            plt.title(class_names[labels[i]])
            plt.axis('off')

train_ds= tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                              labels='inferred',
                                                              label_mode='int',
                                                              class_names=None,
                                                              color_mode='rgb',
                                                              batch_size=batch_size,
                                                              image_size=(img_h,img_w),
                                                              shuffle=True,
                                                              validation_split=.2,
                                                              seed=123,
                                                              subset='training')

validation_ds= tf.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                              labels='inferred',
                                                              label_mode='int',
                                                              class_names=None,
                                                              color_mode='rgb',
                                                              batch_size=batch_size,
                                                              image_size=(img_h,img_w),
                                                              shuffle=True,
                                                              validation_split=.2,
                                                              seed=123,
                                                              subset='validation')


class_names = train_ds.class_names


##########################
## Remove data bottle neck
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

############################
### apply filters to dataset

rescale= tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)

train_ds_normalize=train_ds.map(lambda image,label: (rescale(image),label))

#### However Instead of applying rescale or resizing in dataset, we can apply them in model

model= tf.keras.Sequential(layers=[
    tf.keras.layers.experimental.preprocessing.Rescaling(1/255,input_shape=(img_h,img_w,3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names),activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

results= model.fit(train_ds,validation_data=validation_ds,epochs=3)



a=1