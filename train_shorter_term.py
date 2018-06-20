# Given the statistics for past 3 hours,
# make traffic predictions for the next 4 hours

import numpy as np
import random
import math
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding, LSTM, Dropout, Dense, Flatten
from keras.constraints import max_norm

epochs = 100
batch_size = 200

script_name = 'shorter_term'

model_types = ['conv1d_1', 'dense_1', 'lstm_1']

active_model = 'lstm_1'

source_csv = 'hki_liikennemaarat.csv'
source_csv_delimiter = ';'

stops = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06']

# fix random seed for reproducibility
np.random.seed(7)

source_data = np.genfromtxt(source_csv, delimiter=source_csv_delimiter, skip_header=1, usecols=[0,4,5,7,8,9,10,11,12,13], encoding='iso-8859-1', unpack=True, dtype=['U3', 'U32', 'uint32', 'uint32', 'uint8', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16', 'uint16'])

print(source_data[25000])

traffic_data = []

for i in range(len(source_data)):
  # Only get data on the hour, discard the rest
  if(source_data[i][2] % 100 == 0):
    traffic_data.append([
      stops.index(source_data[i][0]) / len(stops),
      source_data[i][1] - 1,
      source_data[i][2] / 2400,
      source_data[i][3] / 5000,
      source_data[i][4] / 1000,
      source_data[i][5] / 100,
      source_data[i][6] / 250,
      source_data[i][7] / 250,
      source_data[i][8] / 250,
      source_data[i][9] / 50
    ])

print(traffic_data[10000])

# Get training and test data

trainX = []
trainY = []

# Generation: get current (now) traffic data
# and output is the future data in 1 hour,
# 2 hours, 3 hours and 4 hours
for i in range(len(traffic_data)):
  # time has to be in between 3:00 and 19:00
  if(traffic_data[i][2] >= (3/24) and traffic_data[i][2] <= (19/24)):
    trainX.append(traffic_data[i])
    trainY.append([
      traffic_data[i+1][3],
      traffic_data[i+1][4],
      traffic_data[i+1][5],
      traffic_data[i+1][6],
      traffic_data[i+1][7],
      traffic_data[i+1][8],
      traffic_data[i+1][9],
      traffic_data[i+2][3],
      traffic_data[i+2][4],
      traffic_data[i+2][5],
      traffic_data[i+2][6],
      traffic_data[i+2][7],
      traffic_data[i+2][8],
      traffic_data[i+2][9],
      traffic_data[i+3][3],
      traffic_data[i+3][4],
      traffic_data[i+3][5],
      traffic_data[i+3][6],
      traffic_data[i+3][7],
      traffic_data[i+3][8],
      traffic_data[i+3][9],
      traffic_data[i+4][3],
      traffic_data[i+4][4],
      traffic_data[i+4][5],
      traffic_data[i+4][6],
      traffic_data[i+4][7],
      traffic_data[i+4][8],
      traffic_data[i+4][9]
    ])


trainX = np.array(trainX)
trainY = np.array(trainY)

# make it divisable by batch size
remainder = len(trainX) % batch_size
if remainder > 0:
  trainX = trainX[:-remainder]
  trainY = trainY[:-remainder]

print(trainX.shape)
print(trainY.shape)

print("Active model is: " + active_model)

# create and fit model
model = Sequential()

if(active_model == 'conv1d_1'):

  trainX = np.expand_dims(trainX, axis=2)

  model.add(Conv1D(input_shape=(10, 1), filters=200, kernel_size=2, activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Dropout(0.5))
  model.add(Conv1D(filters=500, kernel_size=1, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(28, activation='sigmoid'))

elif(active_model == 'dense_1'):

  model.add(Dense(30, input_shape=(10, ), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(28, activation='sigmoid'))

elif(active_model == 'lstm_1'):

  trainX = np.expand_dims(trainX, axis=2)

  model.add(LSTM(32, input_shape=(10, 1), return_sequences=True, activation='relu'))
  model.add(Dropout(0.5))
  model.add(LSTM(32))
  model.add(Dropout(0.5))
  model.add(Dense(28, activation='sigmoid'))

model.compile(loss='mae', optimizer='adam')

while True:
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
  model.save(script_name + '_' + active_model + '.h5')
