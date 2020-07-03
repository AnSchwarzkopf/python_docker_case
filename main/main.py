import logging

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    return RandomForestClassifier(n_jobs=-1).fit(X_train, y_train)

def predictTestsplit(model, X_train):
    '''Predict model on test split and return predicted Values'''
    logging.info("Predict on test split...")
    return model.predict(X_train)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

    X, y = fetchDatasets()
    # train samples is already provided, test size from http://yann.lecun.com/exdb/mnist/: The training set contains 60000 examples, and the test set 10000 examples.
    X_train, X_test, y_train, y_test = createTrainAndTestSplit(X, y, train_samples=60000, test_size=10000)

    rfmodel = trainModel(X_train, y_train)
    predicted = predictTestsplit(rfmodel, X_train)

    logging.info("Done")