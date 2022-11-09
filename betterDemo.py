from sklearn.impute import *
from sklearn.model_selection import *
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# # ---------------------------------------------------------------------------------------------------
ds = pd.read_csv('Data.csv')
X = ds.iloc[:, :-1]
Y = ds.iloc[:, -1]
print("\nThese are the data splitted into X and Y:\n")
print(X)
print(Y)


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X.iloc[:, 1:] = imp.fit_transform(X.iloc[:, 1:])
print("\nThis is the values of X after filling the missing data with mean:\n")
print(X)


X = pd.get_dummies(X, columns=['Country'])
print("\nThis is the data after encoding the indipendent variables:\n")
print(X)
print(Y)

le = LabelEncoder()
Y = le.fit_transform(Y)
print("\nhese are the values after encoding the dependent variables:\n")
print(X)
print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=4, random_state=50)

print("\nX_train data: \n", X_train,  "\n", "_X_test data:\n ",  X_test,   "\n",
      "\nY_train data:\n",       Y_train,     "\n",      "\nY_test data: \n",   Y_test)

sc = StandardScaler()
X_train.iloc[:, :2] = sc.fit_transform(X_train.iloc[:, :2])
X_test.iloc[:, :2] = sc.fit_transform(X_test.iloc[:, :2])

print("\nThese are the data after Standard scaling: \n")

print(X_train)
print(X_test)
# -----------------------------------------------------------------------------------------
