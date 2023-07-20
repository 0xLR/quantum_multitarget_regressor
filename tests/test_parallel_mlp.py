
import os, sys
data_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(data_path)
import csv
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import multiprocessing

def train_model(n_features, n_targets, n_samples, optimizer, iteration, test_size, X_train, X_test, Y_train, Y_test):
    # n_features, n_targets, n_samples, optimizer, iteration, test_size, X_train, X_test, Y_train, Y_test = arguments

    print(f"Started with {n_features, n_targets, n_samples, optimizer, iteration, test_size}")
    qnn = MLPRegressor(max_iter=iteration, solver=optimizer).fit(X_train, Y_train)
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
    
    mae = mean_absolute_error(y_true=Y_test, y_pred=predictions)
    mse = mean_squared_error(y_true=Y_test, y_pred=predictions, squared=True)
    rmse = mean_squared_error(y_true=Y_test, y_pred=predictions, squared=False)
    
    filename = 'classical_results-2023-07-04.csv'
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
    

if __name__ == "__main__":
    print("Beginning Tests")
    pool = multiprocessing.Pool(processes=4)
    features_targets = [(3,2)]
    optimizers = ['adam', 'lbfgs','sgd']
    sample_sizes = [100, 500, 1000]
    test_sizes = [20, 30, 40]    
    iterations = [100, 1000, 10000, 100000]

    send = True
    results = []
    processes = []
    for n_features, n_targets in features_targets:
        for n_samples in sample_sizes:
            # Get Dataset
            X, Y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, random_state=42)
            scaler = MinMaxScaler()
            for test_size in test_sizes:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
                Y_train = scaler.fit_transform(Y_train)
                Y_test = scaler.fit_transform(Y_test)
                
                for optimizer in optimizers:
                    for iteration in iterations:
                        train_model(n_features, n_targets, n_samples, optimizer, iteration, test_size, X_train, X_test, Y_train, Y_test)
    #result = pool.map(train_model, processes)
    #results.append(result)
    print('Done!')

                           
