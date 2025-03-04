import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *


@dataclass
class DataIngestion:
    raw_data_path: str = raw_data_path  
    test_size: float = test_size  
    random_state: int = random_state 
    data_path: str = data_path 
    traindata_path: str = traindata_path 
    testdata_path: str = testdata_path 
    

    def import_and_split(self):
        logging.info('Data Ingestion component has started...')
        try:
            logging.info('Reading the raw data')
            self.raw_data = pd.read_csv(self.raw_data_path)
            logging.info(f'shape of the raw data : {self.raw_data.shape}')

            logging.info('spliting the raw data into train and test sets ...')
            self.train_data, self.test_data = train_test_split(self.raw_data, test_size=self.test_size, random_state=self.random_state)
            logging.info(f'shape of the train data : {self.train_data.shape}')
            logging.info(f'shape of the test data : {self.test_data.shape}')

            logging.info('Exporting the raw_data, train_data and test_data to artifact folder ...')
            self.raw_data.to_csv(self.data_path, index =False)
            self.train_data.to_csv(self.traindata_path,index =False)
            self.test_data.to_csv(self.testdata_path,index =False)
            logging.info('Exporting to artifact folder completed')
            logging.info('Data Ingestion component has completed!')

        except Exception as e:
            logging.info(f'An Exception has been occured : {USvisaException(e, sys)}')
            raise USvisaException(e, sys)

# Usage:
# data_ingestion = DataIngestion() #create the data ingestion object.
# data_ingestion.import_and_split() #call the import and split function.
