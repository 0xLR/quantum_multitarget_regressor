import numpy as np
from sklearn.datasets import make_regression
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit.opflow import ZFeature
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.datasets import ad_hoc_data

# Generate the regression dataset
X, y = make_regression(n_samples=1000, n_features=3, n_targets=3, noise=0.1)

# Define the quantum feature map and measurement reduction
feature_map = ZZFeatureMap(3, reps=2)
observable = [ZFeature(feature_dim=3, which_qubits=[i]) for i in range(3)]

# Create the QNN
qnn = TwoLayerQNN(3, 3, feature_map, observable)

# Create the VQC and specify the optimizer
optimizer = {'name': 'SPSA', 'maxiter': 100}
vqc = VQC(optimizer, qnn, training_dataset=(X, y))

# Train the VQC using the quantum instance
backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)
vqc.fit(X, y, quantum_instance=quantum_instance)