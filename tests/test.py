from src.multitarget_qnn_regressor import MultiTargetQuantumNNRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor



if __name__ == "__main__":
    print("Beginning Tests")

    sample_sizes = [100, 500, 1000]
    test_sizes = [20, 30, 40]
    optimizers = ['cobyla', 'adam', 'lbfgsb','slsqp']
    iterations = [100, 300, 500]

    # Get Dataset

    # For each sample size


    # For each train-test split size


    # For each optimizer


    # For each iteration


    # Time testing

    # Time prediction


    # R2 Score


    # MAE Score


    # MSE Score


    # RMSE Score


    # Send email if true

