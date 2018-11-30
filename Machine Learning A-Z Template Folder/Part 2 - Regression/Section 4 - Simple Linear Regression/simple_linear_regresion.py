
#simplel linear regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 / 3, random_state = 0)


# fit simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# predict
y_pred = regressor.predict(X_test)


# visualize
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

# visualzie the test reuslt
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()