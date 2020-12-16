import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Split into Training and Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Train model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict

y_pred = regressor.predict(X_test)

# Visualize both training and test

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Training set)')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Test set)')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()

# Make a single prediction

#print(regressor.predict([[]]))

# Getting the final linear regression equation with the values of the coefficients

#print(regressor.coef_)
#print(regressor.intercept_)