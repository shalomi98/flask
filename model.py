#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model
from joblib import load
import numpy as np

# Load your trained LSTM model
model = load_model('my_univariate_lstm_model.keras')

# Assume the scaler is saved as 'my_scaler.joblib'
scaler = load('my_scaler.joblib')

# The create_dataset function from your code
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 5  # Set the look_back value to whatever was used during training

def preprocess_input(raw_data):
    # Apply the same preprocessing as was done during training
    scaled_data = scaler.transform(np.array(raw_data).reshape(-1, 1))
    X, _ = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X

def make_prediction(input_data):
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    # Invert the scaling
    inverted_prediction = scaler.inverse_transform(prediction)
    return inverted_prediction
