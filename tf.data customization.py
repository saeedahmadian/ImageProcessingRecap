import tensorflow as tf
import glob, os, pathlib
from tensorflow.keras.layers.experimental.preprocessing import Rescaling,Resizing

##### Read files inside each folder

datadir = pathlib.Path('flowers')
class_names = [i.name for i in datadir.glob('*')]

