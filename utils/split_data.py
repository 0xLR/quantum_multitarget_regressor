import numpy as np
from sklearn.model_selection import train_test_split

X_full_data_set = np.load('X_full_dataset.npy')
y_full_data_set = np.load('y_full_dataset.npy')

# Training Splits

training_splits = [80, 75, 70, 65, 60, 55, 50]

for split in training_splits:
    train_size = split/100 # float representation of training size
    test = 100-split # integer representation of test size to include in file_name
    X_train, X_test, y_train, y_test = train_test_split(X_full_data_set, y_full_data_set, random_state=1, train_size=train_size)
    np.save(f'X_train_{split}-{test}', X_train, allow_pickle=True)
    np.save(f'X_test_{split}-{test}', X_test, allow_pickle=True)
    np.save(f'y_train_{split}-{test}', y_train, allow_pickle=True)
    np.save(f'y_test_{split}-{test}', y_test, allow_pickle=True)