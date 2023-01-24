# geoguessr-DNN

This repository contains code for training neural networks over Google Street View images with the task of country classification. It has been done as Project Work of the course Machine Learning for Computer Vision at the University of Bologna (report can be read at `report.pdf`).<br>

## Problem description

Previous approaches only use north, east, west and down frames for country classification.<br>
A lot of countries, anyway, contain meta-data in their coverage, meaning that they use different cars or different cameras. This knowledge is often useful to narrow down the possible countries for a specific guess.<br>
The following images shows the Google car with metal bars on top, typical of specific countries like Guatemala or Ghana.

![meta](https://user-images.githubusercontent.com/37805862/214365869-a6db8059-f438-4f07-b0e4-8b012363dcfb.png)

Our hypothesis is that meta-data could improve the metrics, so we perform tests by providing the down frame as well.

## How to obtain dataset

The dataset is obtained by using `create_data.py`. For each sampled location, we get north, east, south, west, down frames.<br>
The neural networks have been tested on 4 countries (Andorra, Canada, Denmark, Ghana), with 1000 location sampled per country (5000 images each).<br>
To obtain imagery for different countries, change the `country_names` list inside `create_data.py` with the desired countries (you can get a list of the countries covered by Streetview [here](https://www.reddit.com/r/geoguessr/comments/ks6chr/full_list_of_all_countries_possible_in_battle/)).<br>
The code may sometimes fail to obtain a valid Streetview location since it randomly samples a position in a given country and searches over a certain radius.

## Neural Networks tested

The general approach is shown in the following figure

![architecture](https://user-images.githubusercontent.com/37805862/214367494-3447700d-b460-404d-8c1c-de68f6296e46.png)


The classifications and softmax scores for each country are fed into a LambdaMart model, which is also provided with the score of a binary classificator Neural Network that detects Google Car meta data (using down frames).<br>
The trained neural networks are

1. A simple convolutional network
2. ResNet50
3. ResNet50 with pre-trained weights (ImageNet)

The results are provided in the following table. The models with the prefix *m* are the ones that use meta-data.

| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
CNN | 0.40 | 0.40 | 0.39 |
mCNN | 0.50 | 0.53 | 0.51 |
ResNet50 | 0.40 | 0.43 | 0.41 |
mResNet50 | 0.55 | 0.55 | 0.56 |
ResNet50 IN | 0.75 | 0.75 | 0.75 |
mResNet50 IN | 0.73 | 0.72 | 0.73 |
