# Logistic Regression

# Importing the libraries
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:/Chandu/n00b/PYTHON/DS/Social_Network.csv")

# Spliting the dataset in independent and dependent variables
X = dataset.loc[:, ['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling to bring the variable in a single scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
logisticregression = LogisticRegression()
logisticregression.fit(X_train, y_train)

# Predicting the Test set results
y_pred = logisticregression.predict(X_test)
print(y_pred)

# lets see the actual and predicted value side by side
y_compare = np.vstack((y_test, y_pred)).T
# actual value on the left side and predicted value on the right hand side
# printing the top 5 values
y_compare[:5, :]

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# print('true negatives (TN): Both, actual and predicted values are false: ', cm[0,0])
# print('true positives (TP): Both, actual and predicted values are true: ', cm[1,1])
# print('false positives (FP): Predicted value is yes but actual is false: ', cm[0,1])
# print('false negative (FN): Predicted value is no but actual is true: ', cm[1,0])

# Visualising the Training set results
X_set, y_set = X_train, y_train
X, Y = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max(
) + 1, step=0.01), np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

probs = logisticregression.predict(
    np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
plt.contourf(X, Y, probs, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(
        ('red', 'green'))(i), label=j, edgecolor="white")

plt.title('Logistic Regression Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
X_set, y_set = X_test, y_test
X, Y = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max(
) + 1, step=0.01), np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

probs = logisticregression.predict(
    np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
plt.contourf(X, Y, probs, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(
	    ('red', 'green'))(i), label=j, edgecolor="white")

plt.title('Logistic Regression Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
