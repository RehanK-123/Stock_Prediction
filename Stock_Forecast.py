import pandas as pd 
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from flask import Flask

m =[]
df = pd.read_csv("Dominos_Stock_Data.csv")
print(df.head())
df = df.drop(columns=["Date","High","Low","Close","Volume"])
forecast_len = 1

df["Predicted"] = df["Adj Close"].shift(-1)
x = np.array(df.drop(columns="Predicted"))[:-forecast_len]
y = np.array(df['Predicted'])[:-forecast_len]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = SVR(kernel='rbf',gamma=0.001,C=11,epsilon=0.55)

model.fit(x_train,y_train)

res = model.predict(x_test)

print(model.score(x_test,y_test))

for i in range(len(x_test)):
    m.append(x_test[i][1])


mtp.scatter(np.array(m),y_test,color="cornflowerblue")
mtp.plot(np.array(m),res,color="red")

mtp.show()


