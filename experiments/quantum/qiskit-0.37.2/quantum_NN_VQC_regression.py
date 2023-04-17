import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.opflow import Z, I
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.neural_networks import CircuitQNN
#from qiskit_machine_learning.utils import split_dataset_to_data_and_labels

X, y = make_regression(n_samples=1000, n_features=3, n_targets=3, noise=0.1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

feature_map = ZZFeatureMap(feature_dimension=3, reps=2)
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=3)

qnn = CircuitQNN(circuit=feature_map.compose(ansatz), input_params=feature_map.parameters, weight_params=ansatz.parameters, interpret= lambda x: x[0])

train_size = int(0.7 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

backend = Aer.get_backend('aer_simulator')
vqc = VQC(optimizer=None, quantum_instance=backend, ansatz=ansatz,feature_map=feature_map)
vqc.fit(X_train, y_train)


y_pred = vqc.predict(X_test)
