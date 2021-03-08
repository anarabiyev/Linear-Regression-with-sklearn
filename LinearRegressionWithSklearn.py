from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')

X = train_data['x']
Y = train_data['y']

X = np.array(X)       
Y = np.array(Y)

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

model = LinearRegression().fit(X, Y)
print(model.coef_)
print(model.intercept_)

test_data = pd.read_csv('test.csv')
X_test = np.array(test_data['x']).reshape(-1, 1)
Y_test = np.array(test_data['y']).reshape(-1, 1)

print(model.score(X_test, Y_test))