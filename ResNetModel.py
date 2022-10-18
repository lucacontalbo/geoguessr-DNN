import tensorflow as tf


class ResNetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet = tf.keras.applications.resnet50.ResNet50(include_top=False,weights=None)
        self.concatenate = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, input):  # input is composed by 6 images
        features1, features2, features3, features4, features5, features6 = self.resnet(input[0]), self.resnet(input[1]), self.resnet(
            input[2]), self.resnet(input[3]), self.resnet(input[4]), self.resnet(input[5])
        concatenated_features = self.concatenate([features1, features2, features3, features4, features5, features6])
        dense1_output = self.dense1(concatenated_features)
        output = self.dense2(dense1_output)
        return output
