import logging

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def fetchDatasets():
    '''Fetch dataset from openml provided from code snipped and returns data and target'''
    
    logging.info("Start fetching datasets...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="/app/scikit")
    return X, y

def createTrainAndTestSplit(X, y, train_samples, test_size):
    '''Split arrays or matrices into train and test subsets and returns data and target train and test sets'''
    
    logging.info("Create train and test split...")
    return train_test_split(X, y, train_size=train_samples, test_size=test_size)

def trainModel(X_train, y_train):
    '''Train model on training split and return trained model'''

    logging.info("Training on train split...")
    return RandomForestClassifier(max_depth=20, max_features=3, min_samples_leaf=3,
                           min_samples_split=8, n_estimators=300, n_jobs=-1).fit(X_train, y_train)

def predictTestsplit(model, X_train):
    '''Predict model on test split and return predicted Values'''

    logging.info("Predict on test split...")
    return model.predict(X_train)

def calculateAccuracy(model, X_train, y_test):
    '''Calculate and print the accuracy of the trained model'''
    
    logging.info("Calculating accuracy of trained model...")
    logging.info("Accuracy: %s" % model.score(X_test, y_test))

def parameterOptimization(model):
    '''Run parameter optimization on model using GridSearchCV and print best estimator'''

    logging.info("Running parameter optimization on model...")
    param_grid = {
        'max_depth': [10, 20],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300]
    }
    grid = GridSearchCV(model, param_grid = param_grid, cv = 3, verbose = 1, n_jobs = -1)
    grid.fit(X_train, y_train)
    logging.info("Best estimator: %s" % grid.best_estimator_)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

    X, y = fetchDatasets()
    # train samples is already provided, test size from http://yann.lecun.com/exdb/mnist/: The training set contains 60000 examples, and the test set 10000 examples.
    X_train, X_test, y_train, y_test = createTrainAndTestSplit(X, y, train_samples=60000, test_size=10000)

    rfmodel = trainModel(X_train, y_train)
    predicted = predictTestsplit(rfmodel, X_train)

    calculateAccuracy(rfmodel, X_test, y_test)

    # parameterOptimization(rfmodel)

    logging.info("Done")