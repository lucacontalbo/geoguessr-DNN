import numpy as np
import tensorflow as tf
from copy import deepcopy
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from models.ResNetModel import ResNetModel
from models.ResNetModelIN import ResNetModelIN
from models.SimpleCNNModel import SimpleCNNModel
from DatasetParser import DatasetParser
import itertools

import random as python_random

np.random.seed(123)
python_random.seed(123)
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
				steps_per_epoch = 11750/fit_info['batch_size']
			else:
				class_weights={0:1,1:1,2:1,3:4}
				xtrain, xval, xtest = dataset_parser.get_state2img(preprocess_function,meta=True)
				steps_per_epoch = 11750/fit_info['batch_size']

			history = model.fit(xtrain,validation_data=xval,**fit_info,steps_per_epoch=steps_per_epoch,class_weight=class_weights)

			if history.history['val_loss'][0] < lowest_val_err:
				print("in")
				lowest_val_err = history.history['val_loss'][0]
				best_model = deepcopy(model)

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
			new_dataset[count].append(prediction[0][0])
			count+=1

		return new_dataset, target

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
	model = ResNetModel(len(dataset_parser.states))
	car_meta_model = ResNetModel(1,'sigmoid')
	preprocess_function = tf.keras.applications.resnet50.preprocess_input
elif MODEL == 'resnetIN':
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

print("--------------------- TRAINING XGBOOST ---------------------")
tp.train_xgboost(best_model,car_meta_model,new_dataset,target,dataset_parser,preprocess_function)

