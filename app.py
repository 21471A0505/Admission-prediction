# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def hello_word():
#     #return "Hello World ....!"
#     return render_template('index.html')
# @app.route('/', method=['POST'])
# def predict():
#     return ""
# if __name__== '__main__':
#     app.run(port=3000, debug=True)






# Importing necessary libraries
from flask import Flask, request, render_template
import joblib
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import sklearn
# Using GridSearchCV to find the best algorithm for this problem


# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    gre = float(request.form['gre'])
    toefl = float(request.form['toefl'])
    university_rating = float(request.form['university_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = float(request.form['research'])
    
    # Make prediction using the model
    prediction = model.predict([[gre, toefl, university_rating, sop, lor, cgpa, research]])
    output = round(prediction[0], 3) * 100  # Convert to percentage
    
    # Render the prediction result to the user
    return render_template('result.html', prediction_text=f'Probability of admission: {output}%')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

