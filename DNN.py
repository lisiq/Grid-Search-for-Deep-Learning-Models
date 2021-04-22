import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, explained_variance_score
from sklearn.utils import shuffle
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import gc



# Load data
X = pd.read_csv('X_train.csv')
X_train = X.iloc[:, :-1]
y_train = X.loc[:, 'target']

# Delete X and call garbage collection
del X
gc.collect()

# Shuffle datasets
X_train, y_train = shuffle(X_train, y_train)

# Val set
X = pd.read_csv('X_val.csv')
X_val = X.iloc[:, :-1]
y_val = X.loc[:, 'target']

# Delete X and call garbage collection
del X
gc.collect()

# Shuffle datasets
X_val, y_val = shuffle(X_val, y_val)


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Model
def build_model(n_hidden=1, n_neurons=512, learning_rate=3e-3, input_shape=[X_train.shape[1]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='selu'))
        model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(lr=learning_rate, clipnorm=1)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Create the wrapper
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# Here you can add all the hyperparamters that you want to tune
param_distribs = {
"n_hidden": [1, 2, 3, 4, 5, 6],
"n_neurons": [50, 100, 150, 200, 250, 300, 350, 400],
"learning_rate": [0.00003, 0.0003, 0.03, 0.3],
}


# Initialize the search
rnd_search_cv = GridSearchCV(keras_reg,
                                   param_distribs, 
                                   cv=5,
                                   n_jobs=-1)

# Fit the model
rnd_search_cv.fit(X_train, y_train, 
                  epochs=600,
                  batch_size=256,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[keras.callbacks.EarlyStopping(verbose=1,patience=15)])


# Load the Test data
X = pd.read_csv('X_test.csv')
X_test = X.iloc[:, :-1]
y_test = X.loc[:, 'target']

# Standardize the test set
X_test = scaler.transform(X_test)


# Remove X, and call garbage collection
del X
gc.collect()

# Check the erros
print('R^2 for Optimized Parameters: {}'.format(r2_score(y_test, rnd_search_cv.predict(X_test))))
print('MSE for Optimized Parameters: {}'.format(mean_squared_error(y_test, rnd_search_cv.predict(X_test))))
print('MAE for Optimized Parameters: {}'.format(mean_absolute_error(y_test, rnd_search_cv.predict(X_test))))
print('EVS for Optimized Parameters: {}'.format(explained_variance_score(rnd_search_cv.predict(X_test), y_test)))


# Best parameters
print('Best parameters: {}'.format(rnd_search_cv.best_params_))

# Save the best regressor
print('Model:')
model = rnd_search_cv.best_estimator_.model

# Summary of the best model
print('Summary:')
print(model.summary())

# Save locally 
model.save('deep_learning_model.h5')
print('Model saved')

