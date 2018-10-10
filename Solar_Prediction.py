# Solar prediction model
# Trenton Griffiths

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
import math
import csv
from datetime import datetime
from tflearn.data_utils import load_csv, to_categorical
import matplotlib.pyplot as plt
import tensorboard

# Graph the stuff in matplotlib.


def main():
	# import the weather data including header row
	DATA_FILE = 'c:/Users/Trent/Documents/School/201840/CS_5890/PythonStuff/Weather_Prediction/weather.csv'
	data, labels = load_csv(DATA_FILE, target_column=11, columns_to_ignore=[0])

	TrainingSetFeatures = data
	TrainingSetFeatures = preProcessData(TrainingSetFeatures)
	TrainingSetLabels = labels
	catagorizeLabels(TrainingSetLabels)
	TrainingSetLabels = to_categorical(TrainingSetLabels, 9)
	
	# Create a test and training set from the data
	net = tflearn.input_data(shape=[None, 12])
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 32)
	net = tflearn.fully_connected(net, 16)
	net = tflearn.fully_connected(net, 9, activation="softmax")
	net = tflearn.regression(net)

	# Define model
	model = tflearn.DNN(net, tensorboard_verbose=3)

	# Start training
	model.fit(TrainingSetFeatures, TrainingSetLabels, n_epoch = 15, validation_set = 0.15, batch_size=12, show_metric=True)	

	'''Differnt architectures and their average accuracy:
	Softmax ~ 0.6015
	Linear ~ 0.1519
	Sigmoid ~ 0.5926
	ReLu ~ 0.3715
	Softplus ~ 0.5436'''


def preProcessData(data):
	tempData = np.zeros((len(data), 12))
	for i in range(len(data)):
		sample = data[i]
		#grab the date element
		dayStr = sample[0]
		dayOfYear = datetime.strptime(dayStr, "%m/%d/%Y").timetuple().tm_yday
		hours = int(sample[1])
		hourVectorReal = math.cos(2*math.pi * (hours/24))
		hourVectorImg = math.sin(2*math.pi * (hours/24))		
		dayVectorReal = math.cos(2*math.pi * (dayOfYear/365))
		dayVectorImg = math.sin(2*math.pi * (dayOfYear/365))
		tempData[i][0] = hourVectorReal 
		tempData[i][1] = hourVectorImg 
		tempData[i][2] = dayVectorReal 
		tempData[i][3] = dayVectorImg
		tempData[i][4] = sample[2]
		tempData[i][5] = sample[3]
		tempData[i][6] = sample[4]
		tempData[i][7] = sample[5]
		tempData[i][8] = sample[6]
		tempData[i][9] = sample[7]
		tempData[i][10] = sample[8]
		tempData[i][11] = sample[9]
	return tempData


def catagorizeLabels(tempLabels):
	# this needs to be adjusted....
	for i in range(len(tempLabels)):
		evSample = float(tempLabels[i])
		if evSample > 4000:
			tempLabels[i] = 4
		elif evSample > 3500:
			tempLabels[i] = 3.5
		elif evSample > 3000:
			tempLabels[i] = 3
		elif evSample > 2500:
			tempLabels[i] = 2.5
		elif evSample > 2000:
			tempLabels[i] = 2
		elif evSample > 1500:
			tempLabels[i] = 1.5
		elif evSample > 1000:
			tempLabels[i] = 1
		elif evSample > 500:
			tempLabels[i] = 0.5
		else:
			tempLabels[i] = 0	


if __name__ == '__main__':
	main()

