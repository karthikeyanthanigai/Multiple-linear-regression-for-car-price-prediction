# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('car.csv')
#this function will provide the descriptive statistics of the dataset.(only int value)
dataset.describe()

import seaborn as sns
#Other methods like Back Propagation/ Forward Propagation can be used. But Correlation Matrix is best for most speedy analysis.
correlation_matrix = dataset.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


#determine X and y variables(for correlation matrix this values is taken as independent variables)
X = dataset.iloc[:, [0,1,3,5]].values
y = dataset.iloc[:, [2]].values

#label encoding for character data
from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])
labelencoder2 = LabelEncoder()
X[:, -1] = labelencoder2.fit_transform(X[:, -1])

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#split
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#ols
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

#predicting the value
y_pred1= lin_reg.predict(X_test)

#r2_score result
from sklearn.metrics import r2_score, mean_squared_error
r_squared1 = r2_score(y_test, y_pred1)
print("Coefficient of Determination using ols method = ",r_squared1)


#SGD
from sklearn.linear_model import SGDRegressor, LinearRegression
regressor = SGDRegressor(max_iter=10000, tol=1e-3)
regressor.fit(X_train, y_train)

#predicting the value
y_pred = regressor.predict(X_test)

#r2_score result
from sklearn.metrics import r2_score, mean_squared_error
r_squared = r2_score(y_test, y_pred)
print("Coefficient of Determination using sgd method = ",r_squared)

