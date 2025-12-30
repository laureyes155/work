0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(0,len(dataset)):
		dataX.append([dataset[i][0],dataset[i][1],dataset[i][2]])
		dataY.append(dataset[i][3])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset
dataframe = read_csv('../datos/spei_codi_spid_mediana.csv', usecols=[1, 2, 3, 4], engine='python')
dataset0 = dataframe.values
'''print(dataset0)
print(type(dataset0))
'''


indicesAEliminar=[]
for i in range(0,  len(dataset0)):
    if (dataset0[i][0] == 'N/E' or dataset0[i][1] == 'N/E' or dataset0[i][2] == 'N/E'
			or dataset0[i][0] == 'nan' or dataset0[i][1] == 'nan' or dataset0[i][2] == 'nan'):
        indicesAEliminar.append(i)


dataset=np.delete(dataset0, indicesAEliminar, axis=0)
for i in range(0,  len(dataset)):
    dataset[i][0]=float(str(dataset[i][0]).replace(",", ""))
for i in range(0, len(dataset)):
	dataset[i][1]=float(str(dataset[i][1]).replace(",", ""))
'''print(dataset)'''
dataset = dataset.astype('float32')

# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))
'''
dataset = scaler.fit_transform(dataset)
'''



# split into train and test sets
train_size = int(len(dataset) * 0.93)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

'''trainX = trainX.reshape(-1, 1)'''
'''testX = testX.reshape(-1, 1)'''
trainY = trainY.reshape(-1, 1)
testY = testY.reshape(-1, 1)

trainX=scaler.fit_transform(trainX) # Fits & Transform
testX=scaler.fit_transform(testX) # Fits & Transform
trainY=scaler.fit_transform(trainY) # Fits & Transform
testY=scaler.fit_transform(testY) # Fits & Transform


for i in range(0,  len(trainX)):
    print(trainX[i][0],trainX[i][1])

for i in range(0,  len(testX)):
    print(testX[i])

for i in range(0, len(trainY)):
	print(trainY[i])

for i in range(0, len(testY)):
	print(testY[i])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 3))
testX = np.reshape(testX, (testX.shape[0], 3))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
'''trainX = trainX.reshape(-1, 1)
testX = testX.reshape(-1, 1)'''
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

trainScoreR2 = r2_score(trainY, trainPredict)
print('Train Score R2: %.2f R2' % (trainScoreR2))
testScoreR2 = r2_score(testY, testPredict)
print('Test Score R2: %.2f R2' % (testScoreR2))
