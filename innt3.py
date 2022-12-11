#Linear regression

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:/Chandu/n00b/PYTHON/Salaries.csv")

X = df.loc[:, ["Level"]]
Y = df.loc[:, ["Salary"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, Y_pred, color="blue")

plt.show()

# Multi Linear regression


df = pd.read_csv("D:/Chandu/n00b/PYTHON/50_Startups.csv")

X = df.loc[:,["Administration","Marketing Spend"]]
Y = df.loc[:, ["Profit"]]


# X = np.column_stack((X1,X2))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, Y_train)

Y_pred = reg.predict(X_test)


def run_experiment(model):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print("R^2 : ", r2_score(Y_test, Y_pred))
    print("MAE :", mean_absolute_error(Y_test, Y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))


model = LinearRegression()
run_experiment(model)
