import pickle
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
import sys
import os
from tqdm import tqdm
from copy import deepcopy
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from models.ResNetModel import ResNetModel
from models.ResNetModelIN import ResNetModelIN
from models.SimpleCNNModel import SimpleCNNModel
from models.EfficientNetB7Model import EfficientNetB7Model
from DatasetParser import DatasetParser
from models.EfficientNet import *
import itertools

import random as python_random

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.or
tf.random.set_seed(1234)


class TrainingProcedure():
	def __init__(self):
		pass

	def start_train(self,model,dataset_parser,preprocess_function,meta=False,epochs=5):
		lowest_val_err = float('inf')
		best_model = None
		for i in range(epochs):
			print("Starting epoch {}".format(i+1))
			if not meta:
				class_weights={0:1,1:1,2:1,3:1}
				xtrain, xval, xtest = dataset_parser.get_state2img(preprocess_function,meta=False)
				steps_per_epoch = 11750/fit_info['batch_size'] #len(xtrain)/fit_info['batch_size']
			else:
				class_weights={0:1,1:1,2:1,3:4}
				xtrain, xval, xtest = dataset_parser.get_state2img(preprocess_function,meta=True)
				steps_per_epoch = 11750/fit_info['batch_size']

			history = model.fit(xtrain,validation_data=xval,**fit_info,steps_per_epoch=steps_per_epoch,class_weight=class_weights)

			if history.history['val_loss'][0] < lowest_val_err:
				print("in")
				lowest_val_err = history.history['val_loss'][0]
				best_model = deepcopy(model)

		y_true = []
		y_pred = []
		y_pred_real = []

		if meta:
			count = 0

			for el in xtest:
				if count == 0:
					count=1
				tmp = np.array(el[0])
				prediction = best_model.predict(tmp,verbose=0)
				y_pred_real.append(prediction[0])
				y_pred.append(1 if prediction[0] > .3 else 0)
				y_true.append(el[1][0])

			print(y_pred_real)
			print(y_pred)
			print(y_true)
		else:
			for el in xtest:
				tmp = np.array(el[0])
				prediction = best_model.predict(tmp,verbose=0)
				y_pred.append(np.argmax(prediction[0]))
				y_true.append(np.argmax(el[1][0]))
			print(y_pred)
			print(y_true)


		print(metrics.classification_report(y_true,y_pred))
		print(metrics.confusion_matrix(y_true,y_pred))

		return best_model

	def predict_val(self,best_model,car_meta_model,dataset_parser,preprocess_function):
		_, xval, _ = dataset_parser.get_state2img(preprocess_function,meta=False)
		_, xval_meta, _ = dataset_parser.get_state2img(preprocess_function,meta=True)
		count = 0
		new_dataset = []
		target = []
		row = []
		for el in xval:
			tmp = np.array(el[0])
			prediction = best_model.predict(tmp,verbose=0)
			print(prediction)
			print(el[1])
			index = np.argmax(prediction[0])
			row.append(int(index))
			row.append(float(prediction[0][index]))
			if (count+1) % 4 == 0:
				new_dataset.append(row)
				target.append(int(np.argmax(el[1][0])))
				row = []
			count+=1
		count = 0

		for el in xval_meta:
			tmp = np.array(el[0])
			prediction = car_meta_model.predict(tmp,verbose=0)
			print(prediction)
			new_dataset[count].append(prediction[0][0])
			count+=1

		return new_dataset, target
	"""
	def train_car_meta(self,model,preprocess_function):
		lowest_val_err = float('inf')
		best_model = None
		for i in range(1):
			print("Starting epoch {}".format(i+1))
			_, _, _, dataset = dataset_parser.get_state2img(preprocess_function)
			xtrain, xval, _ = dataset

			history = model.fit(xtrain,validation_data=xval,**fit_info)

			if history.history['val_loss'][0] < lowest_val_err:
				lowest_val_err = history.history['val_loss'][0]
				best_model = deepcopy(model)
		return best_model
	"""

	def inspect(self,est):
		print("Best parameters set found on train set:")
		print()
		print(est.best_params_)
		print()
		print("Grid scores on train set:")
		print()
		for i in range(len(est.cv_results_['mean_test_score'])):
			print(round(est.cv_results_['mean_test_score'][i],3),"(+/-",round(est.cv_results_['std_test_score'][i],3),") for",est.cv_results_['params'][i])
		print()
		print()

	def baseline_resnet(self,model,dataset_parser,preprocess_function):
		model = start_train(self,model,dataset_parser,preprocess_function)
		_, _, xtest, _ = dataset_parser.get_state2img(preprocess_function)
		predictions, targets = [],[]
		for el in xtest:
			prediction = model.predict(np.array(el[0]),verbose=0)
			index = int(np.argmax(prediction[0]))
			target_index = int(np.argmax(el[1][0]))
			predictions.append(index)
			targets.append(target_index)
		print(metrics.classification_report(targets,predictions))
		print(metrics.confusion_matrix(targets,predictions))


	def train_xgboost(self,best_model,car_meta_model,new_dataset,target,dataset_parser,preprocess_function):
		_, _, xtest = dataset_parser.get_state2img(preprocess_function,meta=False)

		hyperparams = {
			'n_estimators': [10,100,200],
			'max_depth': [3,4,5],
			'learning_rate': [.2,.4,.6,.8],
		}
		objective = 'multi:softmax'
		num_class = 4

		xgb_dataset_test = []
		xgb_target = []
		predictions = []
		test_target = []
		for i,el in enumerate(xtest):
			if i % 5 == 0:
				if i != 0:
					xgb_dataset_test.append(row)
				xgb_target.append(int(np.argmax(el[1][0])))
				row = []
			tmp = el[0]
			if (i+1) % 5 != 0:
				prediction = best_model.predict(np.array(tmp),verbose=0)
				index_pred = np.argmax(prediction[0])
				row.append(index_pred)
				row.append(prediction[0][index_pred])
				predictions.append(index_pred)
				test_target.append(int(np.argmax(el[1][0])))
			else:
				prediction = car_meta_model.predict(np.array(tmp),verbose=0)
				row.append(prediction[0][0])

		xgb_dataset_test.append(row)

		print(xgb_dataset_test)
		print(xgb_target)
		print(new_dataset)
		print(target)

		"""
		classifier = xgb.XGBClassifier(objective=objective,num_class=num_class)
		gs = GridSearchCV(classifier, hyperparams, scoring='f1_macro')
		gs.fit(np.array(new_dataset)[:,:1],np.array(target))
		print("Just one image:")

		self.inspect(gs)

		best_classifier = gs.best_estimator_
		xgb_predictions = best_classifier.predict(np.array(xgb_dataset_test)[:,:-1])

		print(metrics.classification_report(xgb_target,xgb_predictions))
		print(metrics.confusion_matrix(xgb_target,xgb_predictions))
		"""
		print(np.array(new_dataset).shape)
		print(np.array(target).shape)
		print(np.array(xgb_dataset_test).shape)

		classifier = xgb.XGBClassifier(objective=objective,num_class=num_class)
		gs = GridSearchCV(classifier, hyperparams, scoring='f1_macro')
		gs.fit(np.array(new_dataset)[:,:-1],np.array(target))
		print("Without car meta:")

		self.inspect(gs)

		best_classifier = gs.best_estimator_
		xgb_predictions = best_classifier.predict(np.array(xgb_dataset_test)[:,:-1])

		print(metrics.classification_report(xgb_target,xgb_predictions))
		print(metrics.confusion_matrix(xgb_target,xgb_predictions))

		classifier = xgb.XGBClassifier(objective=objective,num_class=num_class)
		gs = GridSearchCV(classifier, hyperparams, scoring='f1_macro')
		gs.fit(np.array(new_dataset),np.array(target))
		print("With car meta:")

		self.inspect(gs)

		best_classifier = gs.best_estimator_
		xgb_predictions = best_classifier.predict(xgb_dataset_test)

		print(metrics.classification_report(xgb_target,xgb_predictions))
		print(metrics.confusion_matrix(xgb_target,xgb_predictions))

MODEL = 'resnet'

fit_info = {
    'verbose': True,
    'epochs': 1,
    'batch_size': 35,
    'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)]
}

compile_info = {
    'optimizer': tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(1e-4,500000,end_learning_rate=1e-6)),
    'loss': 'categorical_crossentropy',
    'metrics': [tf.keras.metrics.CategoricalCrossentropy(label_smoothing=.0)],
}

compile_info_meta = {
	'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-3),
	'loss': 'binary_crossentropy',
	'metrics': [tf.keras.metrics.BinaryCrossentropy()],
}

dataset_parser = DatasetParser()
tp = TrainingProcedure()

if MODEL == 'simplecnn':
	model = SimpleCNNModel(len(dataset_parser.states))
	car_meta_model = SimpleCNNModel(1,'sigmoid')
	preprocess_function = lambda x: x/255
elif MODEL == 'resnet':
	model = ResNetModelIN(len(dataset_parser.states))
	car_meta_model = ResNetModelIN(1,'sigmoid')
	preprocess_function = tf.keras.applications.resnet50.preprocess_input
else:
	model = EfficientNetB7Model(len(dataset_parser.states))
	car_meta_model = EfficientNetB7Model(1,'sigmoid')
	preprocess_function = lambda x: x

model.compile(**compile_info)

car_meta_model.compile(**compile_info_meta)

print("--------------------- TRAINING RESNET MODEL ---------------------")
best_model = tp.start_train(model,dataset_parser,preprocess_function,epochs=5)

print("--------------------- TRAINING CAR META ---------------------")
car_meta_model = tp.start_train(car_meta_model,dataset_parser,preprocess_function,True,epochs=13)

print("Creating xgboost training data...")
new_dataset,target = tp.predict_val(best_model,car_meta_model,dataset_parser,preprocess_function)

#print(new_dataset)

print("--------------------- TRAINING XGBOOST ---------------------")
tp.train_xgboost(best_model,car_meta_model,new_dataset,target,dataset_parser,preprocess_function)

