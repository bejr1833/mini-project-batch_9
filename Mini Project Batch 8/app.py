from flask import Flask, render_template, request
import numpy as np
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model and scaler
model = load_model('attention_lstm_model.h5', compile=False)  # use compile=False if using custom layer like Attention
scaler_X = joblib.load('scaler_X.pkl')  # you need to save your scaler_X
scaler_y = joblib.load('scaler_y.pkl')  # same for scaler_y

seq_length = 1  # sequence length used during training

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        lat = float(request.form['latitude'])
        lon = float(request.form['longitude'])

        # Get current date
        now = datetime.now()
        year, month, day = now.year, now.month, now.day

        # Create input array
        input_features = np.array([[lat, lon, year, month, day]])
        input_scaled = scaler_X.transform(input_features)
        input_reshaped = input_scaled.reshape((1, seq_length, input_scaled.shape[1]))

        pred_scaled = model.predict(input_reshaped)
        pred_magnitude = scaler_y.inverse_transform(pred_scaled)
        prediction = round(pred_magnitude[0][0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
