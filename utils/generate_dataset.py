from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=4, random_state=1)
print(X.shape, y.shape)

np.save('X_full_dataset', X, allow_pickle=True)
np.save('y_full_dataset', y, allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

np.save('X_train', X_train, allow_pickle=True)
np.save('X_test', X_test, allow_pickle=True)
np.save('y_train', y_train, allow_pickle=True)
np.save('y_test', y_test, allow_pickle=True)