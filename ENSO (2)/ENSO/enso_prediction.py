import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers

# Define the softmax function before it's used
def softmax(logits):
    e_x = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):  # Fix __init__ method typo (_init_ -> __init__)
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

file_path = r"C:\Users\saiya\Downloads\Desktop\ENSO\updated_model_lstm.h5"

try:
    model = load_model(file_path, custom_objects={'LSTM': CustomLSTM})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

if model:
    print("\nModel Architecture:")
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    input_shape = (1, 10, 1)
    dummy_input = np.random.random(input_shape)

    print("\nPerforming a prediction on dummy data...")
    try:
        prediction = model.predict(dummy_input)
        print(f"Prediction (raw): {prediction}")

        probabilities = softmax(prediction[0])  # No error since softmax is now defined

        predicted_class = np.argmax(probabilities)
        classes = ['El Niño', 'La Niña', 'Neutral']
        predicted_label = classes[predicted_class]

        probabilities_percentage = probabilities * 100

        print(f"Prediction Probabilities: {probabilities_percentage}")
        print(f"Predicted Class: {predicted_label} with {probabilities_percentage[predicted_class]:.2f}% confidence.")

    except Exception as e:
        print(f"Error during prediction: {e}")
else:
    print("Model could not be loaded. Please check the file path and model format.")
