import logging

def fetchDatasets():
    logging.info("Start fetching datasets...")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

    fetchDatasets()