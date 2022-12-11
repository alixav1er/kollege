from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv("D:/Chandu/n00b/PYTHON/Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Random Forest model on whole dataset

reg = RandomForestRegressor(n_estimators=10, random_state=0)
reg.fit(X, y)

Y_pred = reg.predict(X)

plt.scatter(X,y,color = "red")
plt.plot(X,Y_pred,color = "blue")

plt.show()


