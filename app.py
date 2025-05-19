import pandas as pd
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 

app = Flask(__name__)

df = pd.read_csv("Dominos_Stock_Data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# print(df)
scaler = MinMaxScaler(feature_range=(0, 1))
df["Adj Close"] = scaler.fit_transform(df[["Adj Close"]])

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(df["Adj Close"].values, seq_length)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = ks.models.Sequential([
    ks.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    ks.layers.LSTM(50, activation='relu'),
    ks.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

@app.route("/", methods= ["GET"])
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return '', 204 

@app.route("/home", methods=["POST", "GET"])
def home():
    print("🚀 Request received:", request.method)
    if request.method == "GET":
        return render_template("Home.html", output="")

    if request.method == "POST":
        local_date_input = request.form.get("date")
        print(local_date_input)
        date_input = datetime.strptime(local_date_input, "%Y-%m-%d")
        date_input = date_input.strftime("%Y-%m-%d")

        print("🚀 Request received: pending to process")
        
        if not date_input:
            return render_template("Home.html", output="❌ Please enter a valid date.")

        print(request.form)  
        try:
            date_input = pd.to_datetime(date_input).normalize()
        except Exception as e:
            return render_template("Home.html", output=f"❌ Invalid date format: {e}")

        temp_df = df[df.index <= date_input]
        if len(temp_df) < seq_length:
            return render_template("Home.html", output="❌ Not enough historical data for prediction.")
  
        last_seq = temp_df["Adj Close"].values[-seq_length:].reshape(1, seq_length, 1)
        if last_seq.shape != (1, seq_length, 1):
            return render_template("Home.html", output="❌ Data formatting error, please try a different date.")
    
        predicted_scaled = model.predict(last_seq)[0][0]
    
        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
        return render_template("Home.html", output=f"💰 Predicted Stock Price: {predicted_price:.2f}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
