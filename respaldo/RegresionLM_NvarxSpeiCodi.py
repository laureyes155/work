
# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.linear_model import LinearRegression
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(0,len(dataset)):
		dataX.append([dataset[i][0],dataset[i][1]])
		dataY.append(dataset[i][2])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset
dataframe = read_csv('../datos/Spei_CodiVSTipoDeCambio_2022_2025.csv', usecols=[1, 2, 3], engine='python')
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
print('------------------trainX')
for i in range(0,  len(trainX)):
    print(trainX[i][0],trainX[i][1])

print('------------------testX')
for i in range(0,  len(testX)):
    print(testX[i][0],testX[i][1])

print('------------------trainY')
for i in range(0, len(trainY)):
	print(trainY[i])

print('------------------testY')
for i in range(0, len(testY)):
	print(testY[i])
trainX=scaler.fit_transform(trainX) # Fits & Transform
testX=scaler.fit_transform(testX) # Fits & Transform
trainY=scaler.fit_transform(trainY) # Fits & Transform
testY=scaler.fit_transform(testY) # Fits & Transform

print('------------------trainX')
for i in range(0,  len(trainX)):
    print(trainX[i][0],trainX[i][1])

print('------------------testX')
for i in range(0,  len(testX)):
    print(testX[i][0],testX[i][1])

print('------------------trainY')
for i in range(0, len(trainY)):
	print(trainY[i])

print('------------------testY')
for i in range(0, len(testY)):
	print(testY[i])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 2))
testX = np.reshape(testX, (testX.shape[0], 2))

model = LinearRegression().fit(trainX, trainY)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

'''r_sq = model.score(trainPredict, trainY)
print(f"coefficient of determination: {r_sq}")'''

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