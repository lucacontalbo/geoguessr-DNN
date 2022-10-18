import os
from PIL import Image
from copy import deepcopy
import numpy as np
from tqdm import tqdm

class DatasetParser:
    def __init__(self, dirpath="data/", num_images_per_location=6):
        self.dirpath = dirpath
        self.num_images_per_location = num_images_per_location
        self.states = os.listdir(self.dirpath)
        self.num_images = len(os.listdir(os.path.join(self.dirpath, self.states[0]))) - 1  # -1 for target.csv


    def get_state2img(self, num_samples_val=1, num_samples_test=1):
        def generator_targets(self,state,i):
            target_filepath = os.path.join(self.dirpath, state, "target.csv")
            with open(target_filepath, "r") as f:
                targets = f.readlines()
            return np.array([float(coordinate) for coordinate in targets[i].split(', ')[:2]])


        xtrain_generator = (np.array([np.array(Image.open(os.path.join(self.dirpath,state,"{}.jpg".format(str((i+1)*(j+1)))))) for i in range(self.num_images_per_location)],generator_targets(self,state,j)) \
                            for state in self.states for j in range(int(self.num_images/self.num_images_per_location)-(num_samples_val+num_samples_test)))
        xval_generator = (np.array([np.array(Image.open(os.path.join(self.dirpath,state,"{}.jpg".format(str((i+1)*(j+1)))))) for i in range(self.num_images_per_location)],generator_targets(self,state,j)) \
                          for state in self.states for j in range(int(self.num_images/self.num_images_per_location)-(num_samples_val+num_samples_test),int(self.num_images/self.num_images_per_location)-(num_samples_test)))
        xtest_generator = (np.array([np.array(Image.open(os.path.join(self.dirpath,state,"{}.jpg".format(str((i+1)*(j+1)))))) for i in range(self.num_images_per_location)],generator_targets(self,state,j)) \
                           for state in self.states for j in range(int(self.num_images/self.num_images_per_location)-num_samples_test,int(self.num_images/self.num_images_per_location)))

        return xtrain_generator, xval_generator, xtest_generator

    def get_state2target(self,num_samples_val=1,num_samples_test=1):
        def generator_targets(self,state,i):
            target_filepath = os.path.join(self.dirpath, state, "target.csv")
            with open(target_filepath, "r") as f:
                targets = f.readlines()
            print(i)
            return np.array([float(coordinate) for coordinate in targets[i].split(', ')[:2]])

        train_max = int(self.num_images/self.num_images_per_location)-(num_samples_val+num_samples_test)
        val_max = int(self.num_images/self.num_images_per_location)-num_samples_test
        test_max = int(self.num_images/self.num_images_per_location)

        print("{},{},{}".format(train_max,val_max,test_max))

        ytrain = (generator_targets(self,state,i) for state in self.states for i in range(0,train_max))
        yval = (generator_targets(self,state,i) for state in self.states for i in range(train_max,val_max))
        ytest = (generator_targets(self,state,i) for state in self.states for i in range(val_max,test_max))

        return ytrain,yval,ytest

    def train_val_test_split(self, state2img, state2target, num_samples_val=1, num_samples_test=1):
        xtrain, ytrain, xval, yval, xtest, ytest = np.array([]), np.array([]), np.array([]), \
                                                   np.array([]), np.array([]), np.array([])

        for state, images in tqdm(state2img.items()):
            xtest = np.append(xtest, np.array(images[:num_samples_test]))
            ytest = np.append(ytest, np.array(state2target[state][:num_samples_test]))

            xval = np.append(xval, np.array(images[num_samples_test:num_samples_test+num_samples_val]))
            yval = np.append(yval, np.array(state2target[state][num_samples_test:num_samples_test+num_samples_val]))

            xtrain = np.append(xtrain, np.array(images[num_samples_test+num_samples_val:]))
            ytrain = np.append(ytrain, np.array(state2target[state][num_samples_test+num_samples_val:]))

            print(xval)

        return xtrain, ytrain, xval, yval, xtest, ytest
