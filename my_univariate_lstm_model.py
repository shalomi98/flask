from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

import pandas as pd

# Load the datasets
train_df = pd.read_csv('/Users/shalomifernandes/Downloads/ML_assessment_data-set/A-univariant.csv')
test_df = pd.read_csv('/Users/shalomifernandes/Downloads/ML_assessment_data-set/Univariant-test.csv')

# Assuming train_df is already defined and loaded
# Normalize the training data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_train = scaler.fit_transform(train_df[['A']].values)

# Convert training data to the time series problem format
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 5
trainX, trainY = create_dataset(scaled_train, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# Create the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[EarlyStopping(monitor='loss', patience=10)])

# Make predictions on the training data
trainPredict = model.predict(trainX)

# Invert predictions back to original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverted = scaler.inverse_transform([trainY])

# Calculate root mean squared error for the training data
trainScore = np.sqrt(mean_squared_error(trainY_inverted[0], trainPredict[:,0]))
print(f'Training Data RMSE: {trainScore}')

# Assuming test_df is already defined and loaded
# Normalize the test data
scaled_test = scaler.transform(test_df[['A']].values)

# Convert test data to the time series problem format
testX, testY = create_dataset(scaled_test, look_back)

# Reshape input to be [samples, time steps, features]
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Make predictions on the test data using the trained model from Part 1
testPredict = model.predict(testX)

# Invert predictions back to original scale
testPredict = scaler.inverse_transform(testPredict)
testY_inverted = scaler.inverse_transform([testY])

# Calculate root mean squared error for the test data
testScore = np.sqrt(mean_squared_error(testY_inverted[0], testPredict[:,0]))
print(f'Test Data RMSE: {testScore}')

#saving the model
from joblib import dump

model.save('my_univariate_lstm_model.keras')

# Save the scaler
dump(scaler, 'my_scaler.joblib')