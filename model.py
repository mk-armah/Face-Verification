import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


#constructing the base network


def base_network(model:tf.keras.Model,trainable = False,input_shape = (120,120,3),out_units = 48):
  model.trainable = trainable
  inputs = tf.keras.layers.Input(shape = input_shape,name= "input_layer")
  x = model(inputs)

  x = tf.keras.layers.Dense(units = 128,name = "first_dense")(x)
  outputs = tf.keras.layers.Dense(units = out_units,name= "dense_output")(x)

  model = tf.keras.Model(inputs= inputs, outputs = outputs)
  return model

class euclidean_distance(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super(euclidean_distance,self).__init__()

  def call(self,input_a,input_b):
    sum_square = K.sum(K.square(input_a,input_b),axis= 1,keepdims = True)
    distance = K.sqrt(K.maximum(sum_square,K.epsilon()))

    return distance

if __name__ == "__main__":
  input_shape = (160,160,3)
  mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = input_shape,include_top= False,weights = "imagenet")
  base_network(mobilenet,input_shape = input_shape)


    

















