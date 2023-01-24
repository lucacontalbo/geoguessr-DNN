import tensorflow as tf
import numpy as np

class ResNetModel(tf.keras.Model):
    def __init__(self,len_output,final_activation='softmax'):
        super().__init__()
        self.resnet = tf.keras.applications.resnet50.ResNet50(include_top=True, classes=len_output, weights=None, classifier_activation=final_activation)

    def summary(self):
        print(self.resnet.summary())

    def call(self, input):
        features1 = self.resnet(input)

        return features1
