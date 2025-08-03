# Save as app.py
from flask import Flask, request, jsonify
import joblib
import yfinance as yf
import numpy as np

app = Flask(__name__)
model = joblib.load("your_model.pkl")  # or tf.keras.models.load_model(...)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data['symbol']
    # Get historical data
    hist = yf.download(symbol, period="60d")  # last 60 days as example
    # Feature engineering as per your model
    # Example: get closing prices, scale, shape, etc.
    closing_prices = hist['Close'].values.astype('float32')
    # Prepare features as expected (e.g., last 30 points)
    X_input = closing_prices[-30:].reshape((1, 30, 1))
    preds = model.predict(X_input).flatten()
    # For future sequences, recursively predict or return multiple values

    # For tomorrow's price comparison
    today_price = float(closing_prices[-1])
    tomorrow_price = float(preds[0])
    is_up = tomorrow_price > today_price

    # You can also create multiple-step-ahead predictions for plotting
    # (e.g., append, re-run, or use a sequence forecasting model)

    # Respond in JSON
    return jsonify({
        "predicted_price": tomorrow_price,
        "today_price": today_price,
        "is_up": is_up,
        "future_dates": [],  # Add dates for the plot if available
        "future_prices": preds.tolist(),  # Or build multi-step forecasts
    })

if __name__ == "__main__":
    app.run(debug=True)