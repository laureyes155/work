
# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, numCols=1):
	dataX, dataY= [], []
	for i in range(0,len(dataset)):
		dataX.insert(i,dataset[i][0:numCols-1])
		dataY.insert(i,dataset[i][numCols-1])
	return np.array(dataX), np.array(dataY)


def clean_dataset(dataset0, numCols=1):
	indicesAEliminar = []
	for i in range(0, len(dataset0)):
		for j in range(0, numCols):
			if dataset0[i][j] == 'N/E' or dataset0[i][j] == 'nan' or dataset0[i][j] == '#VALOR!':
				indicesAEliminar.append(i)

	dataset = np.delete(dataset0, indicesAEliminar, axis=0)

	for i in range(0, len(dataset)):
		for j in range(0, numCols):
			dataset[i][j] = float(str(dataset[i][j]).replace(",", ""))
	dataset = dataset.astype('float32')
	return dataset

def clean_dataset_esp(dataset0,x):
	indicesAEliminar = []
	for i in range(0, len(dataset0)):
		if (dataset0[i][x] == 'N/E'	or dataset0[i][x] == 'nan' or dataset0[i][x] == '#VALOR!'):
			indicesAEliminar.append(i)

	dataset = np.delete(dataset0, indicesAEliminar, axis=0)
	for i in range(0, len(dataset)):
		dataset[i][x] = float(str(dataset[i][x]).replace(",", ""))
	dataset = dataset.astype('float32')
	return dataset

def clean_dataset_esp2(dataset0):
	for i in range(0, len(dataset0)):
		dataset0.loc[i]= float( np.array_str(dataset0.loc[i].values).replace(",", "").replace("[", "").replace("]", "").replace("'", "").strip())
	dataset = dataset0.astype('float32')
	return dataset

def run_model(fileNameParam,columnsParam,trainSizeParam,algoritmoParam):
	numCols = len(columnsParam)
	# fix random seed for reproducibility
	tf.random.set_seed(7)
	# load the dataset
	dataframe = read_csv(fileNameParam, usecols=columnsParam, engine='python')
	dataset0 = dataframe.values

	dataset = clean_dataset(dataset0,numCols)

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))

	# split into train and test sets
	train_size = int(len(dataset) * trainSizeParam)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back, numCols)
	testX, testY = create_dataset(test, look_back, numCols)

	trainY = trainY.reshape(-1, 1)
	testY = testY.reshape(-1, 1)

	trainX = scaler.fit_transform(trainX)  # Fits & Transform
	testX = scaler.fit_transform(testX)  # Fits & Transform
	trainY = scaler.fit_transform(trainY)  # Fits & Transform
	testY = scaler.fit_transform(testY)  # Fits & Transform

	for i in range(0, len(trainX)):
		print(trainX[i][0:numCols])

	for i in range(0, len(testX)):
		print(testX[i][0:numCols])

	for i in range(0, len(trainY)):
		print(trainY[i])

	for i in range(0, len(testY)):
		print(testY[i])

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], numCols-1))
	testX = np.reshape(testX, (testX.shape[0], numCols-1))

	match algoritmoParam:
		case "lstm":
			model = Sequential()
			model.add(LSTM(4, input_shape=(1, look_back)))
			model.add(Dense(1))
			model.compile(loss='mean_squared_error', optimizer='adam')
			model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
		case "lm":
			model = LinearRegression().fit(trainX, trainY)
		case "lasso":
			model = Lasso(alpha=0.1)
			model.fit(trainX, trainY)
		case "decisionTree":
			model = DecisionTreeRegressor(max_depth=4, random_state=42)
			model.fit(trainX, trainY)
		case "svr":
			model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
			model.fit(trainX, trainY)
		case "rf":
			model = RandomForestRegressor(n_estimators=100, random_state=42)
			model.fit(trainX, trainY)
		case _:
			return

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	trainPredict = trainPredict.reshape(-1, 1)
	testPredict = testPredict.reshape(-1, 1)
	trainY = trainY.reshape(-1, 1)
	testY = testY.reshape(-1, 1)

	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	for i in range(0, len(testPredict)):
		print(testPredict[i])
	# calculate root mean squared error
	trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = np.sqrt(mean_squared_error(testY, testPredict))
	print('Test Score: %.2f RMSE' % (testScore))

