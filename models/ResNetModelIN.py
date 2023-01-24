import tensorflow as tf
import numpy as np

class ResNetModelIN(tf.keras.Model):
    def __init__(self,len_output,final_activation='softmax'):
        super().__init__()
        self.resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, classifier_activation=final_activation, weights='/public.hpc/micheleluca.contalbo/ml4cv/pretrained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.concatenate = tf.keras.layers.Concatenate(axis=0)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        #self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(len_output, activation=final_activation)

        #regularizer = tf.keras.regularizers.l2(1e-6)

        #for layer in self.resnet.layers:
        #   for attr in ['kernel_regularizer']:
        #      if hasattr(layer, attr):
        #         setattr(layer, attr, regularizer)

    def summary(self):
        print(self.resnet.summary())

    def call(self, input):
        features1 = self.flatten(self.avg_pooling(self.resnet(input)))
        dense1 = self.dense1(features1)
        dense2 = self.dense2(dense1)

        return dense2
