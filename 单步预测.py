import numpy as np
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error,mean_absolute_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import pandas as pd
from keras import optimizers
from keras.layers import Bidirectional
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
from pandas import DataFrame

def split_dataset(data):
	# split into standard weeks
	train, test = data[:-33], data[-33:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/33))
	test = array(split(test, len(test)/33))
	return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	
	# calculate an RMSE score for each day
	mae = mean_absolute_error(actual, predicted)
	scores.append(mae)
	# calculate overall RMSE
	# for row in range(actual):
	# 	for col in range(actual):
	# 		s += np.abs((actual[row, col] - predicted[row, col]))
	score =0
	
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.8f' % s for s in scores])
	print('%s: [%.8f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	print(array(X)[0])
	print(array(y)[0])
	return array(X), array(y)

# train the model
def build_model(train, n_input,neruon,epoch,batchsize,cur_optim):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	print("训练集，测试集的shape")
	print(train_x.shape)
	print(train_y.shape)
	print("训练集，测试集的length")
	print(len(train_x))
	print(len(train_y))
	# define parameters
	print(train_x[1])
	verbose, epochs, batch_size = 0, epoch,batchsize
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	print("n_timesteps = "+ str(n_timesteps))
	print("n_features = " + str(n_features))
	model.add((LSTM(neruon, input_shape=(n_timesteps,1))))
	# model.add(Dense(50))
	model.add(Dense(n_outputs))
	model.compile(loss='mean_squared_error', optimizer=cur_optim)
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data[:, :, np.newaxis]
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:,0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	# print("yhat的value")
	# print(yhat)
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input,neruon,epoch,batchsize,cur_optim):
	# fit model
	model = build_model(train, n_input,neruon,epoch,batchsize,cur_optim)
	test = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))
	train=train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
	# history is a list of weekly data
	history = [x for x in train]
	print("history")
	print(history[1])
	print(len(test))
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	mae = mean_absolute_error(predictions, test)
	print("MAE=")
	print(mae)
	return mae,predictions,test

# load the new file
path= "D:\\停车\\论文\\纯预测\\lstm\\程序\\LSTM\\高新国际.xlsx"
# load the new file
dataset = pd.read_excel(path,infer_datetime_format=True,index_col="index",sheet_name="总表")
dataset=dataset["泊位占有率"]
# split into train and test
train, test = split_dataset(dataset.values)
print("刚开始的train的shape")
print(train.shape)
print(test.shape)
train = train[:,:,np.newaxis]
test=test[:,:,np.newaxis]
# evaluate model and get scores
# n_input = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
neruons=[20,30,40,50,60,70,80,90,100]
epochs=[5,10,15,20,25,30]
# batchsizes=[1,5,10,15,20]
optimi =["adam","sgd","Nadam","RMSprop"]
n_input =1
batchsizes=[1]
writer_predict_data= pd.ExcelWriter('D:\\停车\\论文\\纯预测\\lstm\\程序\\LSTM\\1.xlsx')

for m in range(len(optimi)):
	cur_optim = optimi[m]
	data = DataFrame()
	data["true_Data"] = test[:, :, 0][0]
	for i in range(len(neruons)):
		cur_neruo =neruons[i]
		for j in range(len(epochs)):
			cur_epo = epochs[j]
			for k in range(len(batchsizes)):
				cur_batch = batchsizes[k]
				mae, predict_data, true_data = \
					evaluate_model(train, test, n_input, cur_neruo, cur_epo, cur_batch,cur_optim)
				cur_name = "%s,%s,%s"%(cur_neruo,cur_epo,cur_batch)
				data[cur_name]=predict_data
				days = [x for x in range(33)]
	cur_sheet_name =cur_optim
	data.to_excel(writer_predict_data,sheet_name=cur_sheet_name)



writer_predict_data.save()

			
