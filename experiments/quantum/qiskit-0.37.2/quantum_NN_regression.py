import numpy as np
from sklearn.datasets import make_regression
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
#from qiskit.opflow import ZFeature
from qiskit_machine_learning.algorithms import VQC
#from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.datasets import ad_hoc_data

# Generate the regression dataset
X, y = make_regression(n_samples=1000, n_features=3, n_targets=3, noise=0.1)
train_size = int(0.7 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]



#qc = QuantumCircuit(3, 3)
# Define the quantum feature map and measurement reduction
feature_map = ZZFeatureMap(3, reps=2)
#observable = [ZFeature(feature_dim=3, which_qubits=[i]) for i in range(3)]
ansatz = RealAmplitudes(3)
# Create the QNN
#qnn = TwoLayerQNN(3, 3, feature_map, observable)

# Create the VQC and specify the optimizer
optimizer = {'name': 'SPSA', 'maxiter': 100}


# Train the VQC using the quantum instance
backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)
vqc = VQC(3,feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=30), quantum_instance=quantum_instance)
vqc.fit(X_train, y_train)
results = vqc.predict(X_test)
print(results)