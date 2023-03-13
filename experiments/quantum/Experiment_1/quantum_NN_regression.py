import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR

from typing import Union

from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from IPython.display import clear_output

"""
Qiskit Quantum Neural Network Regressor
https://qiskit.org/documentation/machine-learning/tutorials/02_neural_network_classifier_and_regressor.html
"""

quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=1024)

'''
Begin Quantum Regression
'''


data_set_path = "..\\..\\..\\datasets\\75-25\\"
x_file = "X_train_75-25.npy"
y_file = "y_train_75-25.npy"

x_train = np.load(data_set_path + x_file)
y_train = np.load(data_set_path + y_file)


print(x_train)
print(y_train)

# Here we restrict to regression with an OpflowQNN that returns values in . 
# More complex and also multi-dimensional models could be constructed, also based on CircuitQNN but that exceeds the scope of this tutorial.

# construct simple feature map
param_x = Parameter("x")
feature_map = QuantumCircuit(1, name="fm")
feature_map.ry(param_x, 0)

# construct simple ansatz
param_y = Parameter("y")
ansatz = QuantumCircuit(3, name="vf")
ansatz.ry(param_y, 0)

# construct QNN
regression_opflow_qnn = TwoLayerQNN(3, None, ansatz, quantum_instance=quantum_instance)

# construct the regressor from the neural network
regressor = NeuralNetworkRegressor(
    neural_network=regression_opflow_qnn,
    loss="squared_error",
    optimizer=L_BFGS_B()
)

# create empty array for callback to store evaluations of the objective function
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

# fit to data
regressor.fit(x_train, y_train)

# return to default figsize
plt.rcParams["figure.figsize"] = (6, 4)

# score the result
score = regressor.score(x_train, y_train)
print(score)

# plot data
plt.plot(x_train, y_train, "bo")
# # plot fitted line
# y_ = regressor.predict(X_)
# #plt.plot(X_, y_, "g-")
# #plt.show()
