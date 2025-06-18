from flask import Flask, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)

# Load the ENSO dataset
dataset_path = "ENSO.csv"  # Replace with actual dataset file
df = pd.read_csv(dataset_path)

# Load the trained model
model_path = "updated_model_lstm.h5"
model = load_model(model_path)

# ENSO Class Labels
classes = ['El Niño', 'La Niña', 'Neutral']

# Extract last known ENSO event data from the dataset
df['Date'] = pd.to_datetime(df['Date'])
last_date = df['Date'].max()
last_year = last_date.year

# Determine last El Niño and La Niña occurrences
el_nino_last_year = df[df['ENSO Phase-Intensity'].isin(['WE', 'ME', 'SE'])]['Date'].dt.year.max()
la_nina_last_year = df[df['ENSO Phase-Intensity'].isin(['WL', 'ML', 'SL'])]['Date'].dt.year.max()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Prepare input data (Replace with real data preprocessing)
    input_shape = (1, 10, 1)
    dummy_input = np.random.random(input_shape)

    # Perform prediction
    prediction = model.predict(dummy_input)

    # Convert to probabilities
    probabilities = np.exp(prediction - np.max(prediction))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    # Get predicted class
    predicted_class = np.argmax(probabilities)
    predicted_label = classes[predicted_class]

    # Approximate next ENSO event timing
    if predicted_label == "El Niño":
        predicted_year = el_nino_last_year + 3  # Random within 2-7 years range
        predicted_month_range = "March - June"
    elif predicted_label == "La Niña":
        predicted_year = la_nina_last_year +  2 # Random within 2-7 years range
        predicted_month_range = "November - Febraury"
    else:
        predicted_year = last_year + 1  # Neutral phase can occur anytime
        predicted_month_range = "All Months"
    predicted_year=max(2024,predicted_year)
    
    return render_template('predict.html', 
                        event=predicted_label, 
                        confidence=round(probabilities[0][predicted_class] * 100, 2), 
                        year=predicted_year, 
                        month_range=predicted_month_range)

@app.route('/elnino_analysis')
def elnino_analysis():
    return render_template('elnino.html')

@app.route('/lanina_analysis')
def lanina_analysis():
    return render_template('lanino.html')

@app.route('/elnino_history')
def elnino_history():
    return render_template('elpre.html')

@app.route('/lanina_history')
def lanina_history():
    return render_template('lapre.html')

@app.route('/elnino_agr')
def elnino_agr():
    return render_template('elarg.html')

@app.route('/lanina_agr')
def lanina_agr():
    return render_template('laarg.html')


@app.route('/graph')
def graph():
    
    return render_template('graph.html')
@app.route('/forecast')
def forecast():
    
    return render_template('forecast.html')


if __name__ == '__main__':
    app.run(debug=True)