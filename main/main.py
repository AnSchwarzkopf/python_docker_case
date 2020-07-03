import logging

from sklearn.datasets import fetch_openml

def fetchDatasets():
    '''Fetch dataset from openml provided from code snipped and returns data and target'''
    logging.info("Start fetching datasets...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home="/app/scikit")
    return X, y

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

    X, y = fetchDatasets()

    logging.info("Done")