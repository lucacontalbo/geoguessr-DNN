import tensorflow as tf
import numpy as np

class ResNetModel(tf.keras.Model):
    def __init__(self,len_output,final_activation='softmax'):
        super().__init__()
        self.resnet = tf.keras.applications.resnet50.ResNet50(include_top=True, classes=len_output, weights=None, classifier_activation=final_activation) # weights='/public.hpc/micheleluca.contalbo/ml4cv/pretrained_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.concatenate = tf.keras.layers.Concatenate(axis=0)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(len_output, activation=final_activation)

        regularizer = tf.keras.regularizers.l2(1e-6)

        for layer in self.resnet.layers:
           for attr in ['kernel_regularizer']:
              if hasattr(layer, attr):
                 setattr(layer, attr, regularizer)

    def summary(self):
        print(self.resnet.summary())

    def call(self, input):  # input is composed by 6 images
        #features1, features2, features3, features4, features5, features6 = self.resnet(input[0][np.newaxis,:]), self.resnet(input[1][np.newaxis,:]), self.resnet(
        #    input[2][np.newaxis,:]), self.resnet(input[3][np.newaxis,:]), self.resnet(input[4][np.newaxis,:]), self.resnet(input[5][np.newaxis,:])
        #concatenated_features = self.concatenate([features1, features2, features3, features4, features5, features6])
        features1 = self.resnet(input)
        #features1 = self.dense3(self.dense2(self.dense1(self.flatten(self.avg_pooling(features1)))))
        #features1 = self.dense3(self.flatten(self.avg_pooling(features1)))
#        features2 = self.resnet(input[1][np.newaxis,:])
#        features3 = self.resnet(input[2][np.newaxis,:])
#        features4 = self.resnet(input[3][np.newaxis,:])
#        features5 = self.resnet(input[4][np.newaxis,:])
#        features6 = self.resnet(input[5][np.newaxis,:])

#        concatenated_features = self.concatenate([features1, features2, features3, features4])
#        print(concatenated_features.shape)

        #features1 = self.flatten(self.avg_pooling(features1))
#        features2 = self.flatten(self.avg_pooling(features2))
#        features3 = self.flatten(self.avg_pooling(features3))
#        features4 = self.flatten(self.avg_pooling(features4))
#        features5 = self.flatten(self.avg_pooling(features5))
#        features6 = self.flatten(self.avg_pooling(features6))



        #output = self.dense1(features1)
        return features1
