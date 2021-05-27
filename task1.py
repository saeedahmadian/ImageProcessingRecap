import tensorflow as tf
from tensorflow.keras import datasets

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
    def __init__(self,num_classes=10):
        super(Pred,self).__init__()
        self.num_classes = num_classes

        self.layer1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),activation='relu',
                                         strides=(1,1),padding='same',name='conv1_layer',
                                            input_shape=(32,32,3))
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name='max_pooling_1')

        self.layer2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',
                                         strides=(1,1),padding='same', name='conv2_layer')

        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name='max_pooling_2')

        self.layer3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),activation='relu',
                                         strides=(1,1), padding='same', name='conv3_layer')

        self.flat = tf.keras.layers.Flatten()

        self.dense= tf.keras.layers.Dense(64, activation='relu')

        self.out =  tf.keras.layers.Dense(self.num_classes,activation='softmax')

    @tf.function
    def call(self,x_image):
        x = self.layer1(x_image)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.flat(x)
        x = self.dense(x)
        return self.out(x)

    def get_config(self):
        config= super(Pred, self).get_config()
        config.update({'num_class':self.num_classes})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)





train = False
test = True

if train==True:
    convmodel= Pred(10)

    convmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(name='CrossEntropy'),
                           'accuracy']
                  )

    results=convmodel.fit(x=train_images,y=train_labels,batch_size=64,epochs=1,
                          validation_data=(test_images,test_labels),
                          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs'),
                                     tf.keras.callbacks.ModelCheckpoint('checkpoints')])


if test==True:
    newmodel= tf.keras.models.load_model('checkpoints')
    y_pred=tf.argmax(input=newmodel(train_images[0:5,...]),axis=-1).numpy()
    print('Predicted values : {}'.format(y_pred))
    y_true=train_labels[0:5,]
    print('True values : {}'.format(y_true))


## The results shows we are doing good
