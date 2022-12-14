from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("D:/Chandu/n00b/PYTHON/Social_Network.csv")

X = df.loc[:, ["Age", "EstimatedSalary"]].values
Y = df.loc[:, ["Purchased"]].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classi = LogisticRegression()
classi.fit(X_train, Y_train)

print("yo\n", classi.predict(sc.transform([[30, 87000]])))

Y_pred = classi.predict(X_test)


Y_pred = np.reshape(Y_pred, (100, 1))

print(np.size(Y_pred))

print(Y_pred)
print(Y_test)
print(np.concatenate((Y_test, Y_pred), axis=1))

# print(np.concatenate((Y_pred, Y_test.reshape(len(Y_test), 1)), 1))


em = confusion_matrix(Y_test, Y_pred)
print(em)
accuracy_score(Y_test, Y_pred)
