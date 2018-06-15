import numpy as np
import random
import math
from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, Dense, Flatten
from keras.constraints import max_norm

epochs = 10
batch_size = 4

model_types = ['dense_1', 'conv1d_1']

active_model = 'conv1d_1'

source_csv = 'hki_liikennemaarat.csv'
source_csv_delimiter = ';'
source_csv_start_year = 2011
source_csv_end_year = 2017

stops = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06']

# fix random seed for reproducibility
np.random.seed(7)

source_data = np.genfromtxt(source_csv, delimiter=source_csv_delimiter, skip_header=1, usecols=[0,4,5,6,7,8,9,10,11,12,13], encoding='iso-8859-1', unpack=True, dtype=['U3', 'U32', 'uint32', 'uint32', 'uint8', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16'])

print(source_data[25000])

traffic_data = []

for i in range(len(source_data)):
  hours, minutes = divmod(source_data[i][2], 100)
  traffic_data.append([
    stops.index(source_data[i][0]) / len(stops),
    source_data[i][1] - 1,
    (hours + (minutes / 60)) / 24,
    (source_data[i][3] - source_csv_start_year) / (source_csv_end_year - source_csv_start_year),
    source_data[i][4] / 5000,
    source_data[i][5] / 1000,
    source_data[i][6] / 100,
    source_data[i][7] / 250,
    source_data[i][8] / 250,
    source_data[i][9] / 250,
    source_data[i][10] / 50
  ])

print(traffic_data[25000])

# Get training and test data

trainX = []
trainY = []
testX = []
testY = []

for i in range(len(traffic_data)):
  # latest year data is used for testing
  if(traffic_data[i][3] == 1):
    testX.append(traffic_data[i][0:4])
    testY.append(traffic_data[i][4:])
  else:
    trainX.append(traffic_data[i][0:4])
    trainY.append(traffic_data[i][4:])


trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# create and fit model
model = Sequential()

if(active_model == 'dense_1'):

  model.add(Dense(4, input_shape=(4, ), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(12, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation='sigmoid'))

elif(active_model == 'conv1d_1'):

  model.add(Embedding(2000, 50, input_length=4))
  model.add(Dropout(0.3))
  model.add(Conv1D(200, kernel_size=4, padding='valid', activation='relu', strides=1))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(28, activation='relu'))
  model.add(Dropout(0.3))
  # We project onto a single unit output layer, and squash it with a sigmoid:
  model.add(Dense(7, activation='sigmoid'))

model.compile(loss='mae', optimizer='adam')

model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=2)

model.save(active_model+'.h5')