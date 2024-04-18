#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from flask import Flask, request, jsonify
from model import make_prediction

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data expected as a list of values
    data = request.get_json()
    prediction = make_prediction(data['values'])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
