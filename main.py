import pickle
import numpy as np
import tensorflow as tf
import sys
import os
from tqdm import tqdm
from ResNetModel import ResNetModel
from DatasetParser import DatasetParser

fit_info = {
    "verbose": True,
    "epochs": 10,
    "batch_size": 32
}

compile_info = {
    'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-4),
    'loss': 'mean_squared_error',
    'metrics': [tf.keras.metrics.MeanSquaredError()],
}

dataset_parser = DatasetParser()

xtrain, xval, xtest = dataset_parser.get_state2img()
ytrain, yval, ytest = dataset_parser.get_state2target()

#for el in xval:
#	print(el)
#for el in yval:
#	print(el)

print("Starting fit...")
resnet_model = ResNetModel()

resnet_model.compile(**compile_info)

resnet_model.fit(xtrain,ytrain,**fit_info) #validation_data=[xval,yval],**fit_info)

