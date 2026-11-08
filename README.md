![Conceptual illustration of data flowing through a quantum circuit into regression outputs](docs/assets/quantum-regression.png)

# Quantum Multitarget Regressor

A research project exploring quantum neural networks for predicting several continuous values from the same inputs. Built with Qiskit, the model was designed to run on IBM's cloud quantum resources or simulate quantum circuits on a classical computer.

Running experiments on quantum hardware proved expensive, and the available credits ran out. The experimental results in this repository therefore come from simulations on an Azure VM, a classical computer. They describe simulated quantum models and classical baselines, rather than measurements from a quantum processor.

## How the model works

**Multitarget regression** means predicting a vector of numbers instead of a single value. Given an input with $d$ features, the model predicts $m$ targets:

$$
\mathbf{x}\in\mathbb{R}^{d}\quad\longmapsto\quad
\hat{\mathbf{y}}=(f_1(\mathbf{x}),\ldots,f_m(\mathbf{x})).
$$

This implementation trains a separate quantum neural network for each target. All networks receive the same inputs, but each learns its own parameters; there is no shared trainable layer between targets.

### Encoding data into qubits

A **qubit** has a state described by amplitudes, whose squared magnitudes give measurement probabilities. A **feature map** turns ordinary numbers into a quantum state. Here, each feature controls an $R_y$ rotation on its own qubit:

$$
R_y(x_k)|0\rangle
=\cos(x_k/2)|0\rangle+\sin(x_k/2)|1\rangle.
$$

The model uses one qubit per input feature. The ket notation $|0\rangle$ and $|1\rangle$ names the two computational basis states. The `architecture/` notebooks also explore alternative encodings, including ZZ feature maps; the regressor itself uses these simpler rotations.

### Learning a circuit

An **ansatz** is the circuit structure whose adjustable gate angles act as model weights. The regressor uses Qiskit's `RealAmplitudes`, with rotation layers and entangling gates, repeated according to the number of features. Entangling gates allow joint qubit states that cannot always be separated into independent states.

For target $j$, the feature map $U_\phi$ and trainable circuit $U$ prepare:

$$
|\psi_j(\mathbf{x})\rangle
=U(\boldsymbol{\theta}_j)U_\phi(\mathbf{x})|0\rangle^{\otimes d}.
$$

An **observable** is a measurable quantity. `EstimatorQNN` converts the circuit state into a numerical prediction through its expectation value:

$$
f_j(\mathbf{x})
=\langle\psi_j(\mathbf{x})|O|\psi_j(\mathbf{x})\rangle,
\qquad O=Z^{\otimes d}.
$$

Here, $Z$ is the Pauli-Z observable and $\otimes$ combines operators across qubits. Its expectation is the probability of an even number of measured ones minus the probability of an odd number, so the output lies in $[-1,1]$. This is the default observable used by the model's `EstimatorQNN`.

See the official documentation for [EstimatorQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html) and [RealAmplitudes](https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.circuit.library.RealAmplitudes) for more detail; their current APIs differ from the versions used here.

### Training and evaluation

The quantum circuit supplies predictions, while a classical optimizer adjusts its angles to reduce squared error. For each target, the objective can be written as:

$$
\mathcal{L}_j(\boldsymbol{\theta}_j)
=\frac{1}{N}\sum_{i=1}^{N}\left(f_j(\mathbf{x}_i)-y_{ij}\right)^2.
$$

This alternating circuit evaluation and classical optimization is often called **variational quantum learning**. The experiments compare COBYLA, ADAM, L-BFGS-B, and SLSQP, along with classical `MLPRegressor` baselines. Recorded metrics include training and prediction time, MAE, MSE, RMSE, and $R^2$. For a single target:

$$
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(y_i-\hat y_i)^2},
\qquad R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

Lower RMSE indicates smaller errors in the evaluated target's units. An $R^2$ of 1 indicates perfect predictions; 0 matches a constant mean predictor, and negative values are possible. The regressor averages $R^2$ across targets.

## Where the experiments ran

The recorded environment was an Azure **Standard_F16s_v2** VM with 16 vCPUs, 32 GB RAM, and a 128 GB disk. With `use_quantum_cloud=False`, the code evaluates circuits using Qiskit's local `Estimator`. A simulator calculates quantum behavior using classical computation; it does not turn the VM into a quantum computer.

The cloud path uses IBM Quantum Runtime and requests 1,024 **shots** per evaluation. A shot is one circuit execution producing a measurement outcome; repeated shots estimate an expectation value. That cloud configuration is separate from the local simulation used for the reported results. These experiments do not establish a hardware speedup or quantum advantage.

## Running locally

This repository retains its 2023 dependency pins, including Qiskit 0.43.0 and Qiskit Machine Learning 0.6.1. These PowerShell commands create an isolated Python 3.10 environment and install them:

```powershell
py -3.10 -m venv python_env
.\python_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements include Windows-specific packages. The IBM cloud path uses historical Runtime APIs and backend names, so it needs updating before use with a current IBM service.

For a small local example, run this Python code from the repository root:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.multitarget_qnn_regressor import MultiTargetQuantumNNRegressor

X, y = make_regression(
    n_samples=40, n_features=3, n_targets=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Fit preprocessing on training data and reuse it for held-out data.
x_scaler = MinMaxScaler().fit(X_train)
y_scaler = MinMaxScaler().fit(y_train)
X_train = x_scaler.transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.transform(y_train)

model = MultiTargetQuantumNNRegressor(
    n_features=3,
    n_targets=2,
    optimizer="cobyla",
    maxiter=20,
    use_quantum_cloud=False,
)
model.fit(X_train, y_train)
predictions = y_scaler.inverse_transform(model.predict(X_test))
print(predictions)
```

Scaling keeps feature angles and target values in a manageable range. Keep the fitted scalers to transform future inputs and restore prediction units.

The scripts in `tests/` are experiment drivers, including potentially lengthy parameter sweeps. The original drivers scale training and test data separately; for new experiments, fit scalers only on the training split and reuse them as above. Keep this difference in mind when interpreting the archived metrics.

## Repository guide

| Directory | Contents |
| --- | --- |
| `src/` | Multitarget regressor and model persistence helpers |
| `architecture/` | Circuit and feature-map exploration notebooks |
| `tests/` | Quantum simulation and classical baseline experiment scripts |
| `results/` | Archived CSV results, analysis notebooks, and HTML plots |

The header image is an AI-generated conceptual illustration, not an experimental result.
