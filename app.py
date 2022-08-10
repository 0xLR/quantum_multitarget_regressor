from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=4, random_state=1)
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
neural_network = MLPRegressor(max_iter=10000).fit(X_train, y_train)

print("First row in X_train")
print(X_train[0])
print("First row in y_train")
print(y_train[0])

x_to_predict = X_train[0]
print(x_to_predict.shape)
x_to_predict = x_to_predict.reshape(1,-1)
print(x_to_predict.shape)

prediction = neural_network.predict(x_to_predict)
print("Predicted:")
print(prediction)

print(neural_network.score(X_test, y_test))

