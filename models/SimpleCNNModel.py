import tensorflow as tf
import numpy as np

class SimpleCNNModel(tf.keras.Model):
    def __init__(self,output_len,final_activation='softmax'):
        super().__init__()
        info_initializers = {
            'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

        self.sequential = tf.keras.Sequential([
                     tf.keras.layers.Conv2D(16,7,strides=(2,2),activation='relu', **info_initializers),
                     tf.keras.layers.Conv2D(32,5,strides=(2,2),activation='relu', **info_initializers),
                     tf.keras.layers.Dropout(.3),
                     tf.keras.layers.Conv2D(64,5,strides=(2,2),activation='relu', **info_initializers),
                     tf.keras.layers.Conv2D(128,3,strides=(2,2),activation='relu', **info_initializers),
                     tf.keras.layers.Dropout(.3),
                     tf.keras.layers.Conv2D(256,3,strides=(2,2),activation='relu', **info_initializers),
                     tf.keras.layers.Conv2D(512,3,activation='relu', **info_initializers),
                     tf.keras.layers.Dropout(.3),
                     tf.keras.layers.Conv2D(512,3,activation='relu', **info_initializers),
                     tf.keras.layers.GlobalAveragePooling2D(),
                     tf.keras.layers.Flatten()
        ])

        self.concatenate = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(256,activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_len, activation=final_activation)

    def call(self, input):  # input is composed by 5 images
        conv1 = self.sequential(input)
        dense1_output = self.dense1(conv1)
        output = self.dense2(dense1_output)
        return output
