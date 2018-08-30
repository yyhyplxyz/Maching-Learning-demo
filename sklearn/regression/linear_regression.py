import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#The following is simple single value linear-regression
# data = pd.read_csv('Salary_Data.csv')
#
# x = data.iloc[:,:-1].values
# y = data.iloc[:,:1].values
#
# from sklearn.cross_validation import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 1/3)
#
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(x_train,y_train)
#
# y_predict = regressor.predict(x_test)
# plt.scatter(x_train,y_train,color='r')
# plt.plot(x_train,regressor.predict(x_train),color='b')
# plt.show()
#
#
# plt.scatter(x_test,y_test,color='r')
# plt.plot(x_train,regressor.predict(x_train),color='b')
# plt.show()

#The following is simple multi-value linear regression
import os

print(os.path.realpath(__file__))

dataset = pd.read_csv('regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

