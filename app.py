import os 
import pandas as pd
import numpy as np
import keras as ks
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 

app = Flask(__name__)
df = pd.read_csv("Dominos_Stock_Data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
with open('scaler_pkl', 'rb') as f:
    scaler = pickle.load(f)
df["Adj Close"] = scaler.fit_transform(df[["Adj Close"]])

@app.route("/", methods= ["GET"])
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return '', 204 

@app.route("/home", methods=["POST", "GET"])
def home():
    print(f"\n🌐 Received {request.method} request at {datetime.now()}")
    print(f"📦 Request form data: {request.form}")
    return render_template("Home.html")

@app.route("/result", methods= ["POST", "GET"])
def result():
    date = request.form.get("date")
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M")
    date = date.strftime("%Y-%m-%d")
    date = pd.to_datetime(date)
    temp_df = df[df.index <= date]
    seq_length = 10
    print(temp_df.head())
    # if len(temp_df) < seq_length:
    #     print(f"❌ Insufficient data (have {len(temp_df)}, need {seq_length})")
    last_seq = temp_df["Adj Close"].values[-seq_length:].reshape(1, seq_length, 1)
    with open('model_pkl' , 'rb') as f:
        model = pickle.load(f)
    print(last_seq.shape, last_seq)
    print(model)
    predicted_scaled = model.predict(last_seq)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
        
    # print(f"✅ Prediction: {predicted_price:.2f}")
    #"💰 Predicted Stock Price: {predicted_price:.2f}
    return render_template("result.html", output= int(predicted_price))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug= True)
