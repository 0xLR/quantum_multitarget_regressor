
import os, sys
import csv
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from src.multitarget_qnn_regressor import MultiTargetQuantumNNRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import time



if __name__ == "__main__":
    print("Beginning Tests")

    features_targets = [(3,2)]
    optimizers = ['cobyla', 'adam', 'lbfgsb','slsqp']
    sample_sizes = [100, 500, 1000]
    test_sizes = [20, 30, 40]    
    iterations = [100, 300, 500]

    send = True
    
    for n_features, n_targets in features_targets:
        for n_samples in sample_sizes:
            # Get Dataset
            X, Y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, random_state=42)
            
            for test_size in test_sizes:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
                X_train, Y_train = MultiTargetQuantumNNRegressor.get_scaled_data(X_train, Y_train)
                X_test, Y_test = MultiTargetQuantumNNRegressor.get_scaled_data(X_test, Y_test)
                
                for optimizer in optimizers:
                    for iteration in iterations:
                        qnn = MultiTargetQuantumNNRegressor(
                            n_features=n_features, 
                            n_targets=n_targets,
                            optimizer=optimizer,
                            maxiter=iteration,
                            use_quantum_cloud=False,
                            backend=None
                            )
                        # Start training timer
                        start = time.time()
                        qnn.fit(X_train, Y_train)
                        end = time.time()
                        
                        training_time = end-start
                        
                        r2 = qnn.score(X_test, Y_test)
                        
                        start = time.time()
                        predictions = qnn.predict(X_test)
                        end = time.time()
                        
                        prediction_time = end-start
                        
                        mae = qnn.mae(Y_test, predictions)
                        mse = qnn.mse(Y_test, predictions)
                        rmse = qnn.rmse(Y_test, predictions)
                        
                        filename = 'results.csv'
                        file_exists = os.path.isfile(filename)
                        with open (filename, 'a') as csvfile:
                            headers = ['n_features', 'n_targets', 'n_samples', 'test_size', 'optimizer', 'iterations', 'training_time', 'prediction_time', 'r2', 'mae', 'mse', 'rmse']
                            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

                            if not file_exists:
                                writer.writeheader()  # file doesn't exist yet, write a header

                            final_result = {'n_features': n_features,
                                              'n_targets': n_targets,
                                                'n_samples': n_samples,
                                                  'test_size': test_size,
                                                    'optimizer': optimizer,
                                                      'iterations': iteration,
                                                        'training_time': training_time,
                                                          'prediction_time': prediction_time,
                                                            'r2': r2,
                                                              'mae': mae,
                                                                'mse':mse,
                                                                  'rmse': rmse}
                            writer.writerow(final_result)
