import os
from PIL import Image
from copy import deepcopy
import numpy as np
from tqdm import tqdm

CAR_META = ['Senegal','Ghana','Guatemala','Kyrgyzstan','Kenya','Laos']

class DatasetParser:
    def __init__(self, dirpath="data/nesw/", num_images_per_location=5):
        self.dirpath = dirpath
        self.num_images_per_location = num_images_per_location
        self.states = os.listdir(self.dirpath)
        self.num_images = len(os.listdir(os.path.join(self.dirpath, self.states[0]))) - 2  # -1 for target.csv

    def get_state_from_index(self,idx):
        return self.states[idx]

    def get_state2img(self,preprocess_function,meta,num_samples_val=20,num_samples_test=10):
        def generator_samples(self,range_min,range_max,meta,type_data='train'):
            if type_data == 'test':
                num_images = 5
            elif meta:
                num_images = 1
            else:
                num_images = 4

            for i in range(range_min,range_max,self.num_images_per_location):
                for s in range(len(self.states)):
                    tmp = []
                    if meta:
                        if self.states[s] in CAR_META: target = np.array([1])
                        else: target = np.array([0])
                    else:
                        target = np.zeros((1,len(self.states)))
                        target[0][s] = 1

                    for j in range(num_images):
                        if meta:
                              path = os.path.join(self.dirpath,self.states[s],"{}.jpg".format(str(i+self.num_images_per_location)))
                        else:
                              path = os.path.join(self.dirpath,self.states[s],"{}.jpg".format(str(i+j+1)))
                        tmp = preprocess_function(np.array(Image.open(path)))[np.newaxis,:]
                        yield [tmp,target]

        xtrain_max = self.num_images-(num_samples_val+num_samples_test)*self.num_images_per_location
        xval_max = self.num_images-num_samples_test*self.num_images_per_location

        xtrain_generator = generator_samples(self,0,xtrain_max,meta)
        xval_generator = generator_samples(self,xtrain_max,xval_max,meta,type_data='val')
        xtest_generator = generator_samples(self,xval_max,self.num_images-1,meta,type_data='test')

        return xtrain_generator, xval_generator, xtest_generator

