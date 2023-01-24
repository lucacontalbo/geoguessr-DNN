import tensorflow as tf
import numpy as np
from models.EfficientNet import *

class EfficientNetB7Model(tf.keras.Model):
    def __init__(self,output_len):
        super().__init__()
        #self.efficientnet = tf.keras.applications.EfficientNetB4(include_top=False,input_shape=(640,640,3), weights=None) 
#weights='/public.hpc/micheleluca.contalbo/ml4cv/pretrained_weights/efficientnet-b4_imagenet_1000_notop.h5')

        self.efficientnet = EfficientNetB3(include_top=True,classes=output_len,input_shape=(640,640,3))
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.concatenate = tf.keras.layers.Concatenate(axis=0)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        #self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(len_output, activation=final_activation)

    def summary(self):
        print(self.efficientnet.summary())

    def call(self, input):
        features1 = self.flatten(self.avg_pooling(self.efficientnet(input)))
        #dense1 = self.dense1(features1)
        #dense2 = self.dense2(dense1)

        return features1
