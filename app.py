import pandas as pd 
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from flask import Flask,render_template,redirect,request,url_for
date = None
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/home",methods=["GET","POST"])
def home():
    #render_template('Home.html')
    date = request.form.get("date",default="2020-01-01")
    print(date)
    date = pd.to_datetime(date)
    m =[]
    df = pd.read_csv("Dominos_Stock_Data.csv")

    temp_df = df[pd.to_datetime(df['Date'])<=date]


    temp_df = temp_df.drop(columns=["Date","High","Low","Close","Volume"])
    forecast_len = 1

    temp_df["Predicted"] = temp_df["Adj Close"].shift(-1)

    x = np.array(temp_df.drop(columns="Predicted"))[:-forecast_len]
    y = np.array(temp_df['Predicted'])[:-forecast_len]

    x_train = x[:-1]
    y_train = y[:-1]

    x_test = x[-1].reshape(1,-1)  #used to reshape the 1-D array into 2-D
    y_test = y[-1].reshape(-1,1)  #used to reshape the single feature into a 1-D array

    #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    model = SVR(kernel='rbf',gamma=0.001,C=11,epsilon=0.55)
    model.fit(x_train,y_train) #trains the model 

    #model.score(x_test,y_test) "Gives the score when the train_test_split function is used"

    res = model.predict(x_test) #predicts the value using non-linear vector regression
    
    print(res[0])

    for i in range(len(x_test)):
        m.append(x_test[i][1])



    #mtp.scatter(np.array(m),y_test,color="green")
    #mtp.plot(np.array(m),res,color="red")

    #mtp.show()

    return render_template("Home.html", output = res[0])


if __name__ == "__main__":
    app.run()