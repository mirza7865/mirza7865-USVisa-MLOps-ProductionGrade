import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
import yaml

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *


@dataclass
class DataValidation:
    
    data_path:str
    traindata_path:str
    testdata_path:str
    schema_file_path:str
    
    logging.info("Data Validation component has started ...")
    
    def load_data(self):
        try:
            logging.info("Reading data from artifact folder")
            self.raw_df  = pd.read_csv(data_path)
            self.train_df  = pd.read_csv(traindata_path)
            self.test_df  = pd.read_csv(testdata_path)
            logging.info("loading data from artifact folder succesfully!")
            # return self.raw_df, self.train_df,self.test_df
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
    
    def load_schema(self):
        try:
            logging.info("Reading schema file from config folder")
            with open(schema_file_path, 'r') as file:
                self.schema = yaml.safe_load(file)
            logging.info("schema file loaded succesfully!")
            # return self.schema
        
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
    
    def validate_columns(self):
        
        logging.info("Validating columns started ...")
        
        try:
        
            schema_columns = [list(col.keys())[0] for col in self.schema['columns']]
            
            if len(self.raw_df.columns) != len(schema_columns):
                logging.error(f"column mismatch occured, schema columns : {len(schema_columns)}, data columns : {len(self.raw_df.columns)}")
            
            for column in schema_columns:
                if column not in list(self.raw_df.columns):
                    logging.error(f"columns:{column}, not found in the data")
                    
            logging.info("columns validated succesfully!")
        
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
        
    def validate_numerical_columns(self):
        
        logging.info("Validating numerical columns started ...")
        
        try:
        
            data_numerical_columns = list(self.raw_df.select_dtypes(include=['number']))
            schema_numerical_columns = list(self.schema['numerical_columns'])
            
            if len(data_numerical_columns) != len(schema_numerical_columns):
                logging.error(f"column mismatch occured, schema columns has: {len(schema_numerical_columns)} columns, data columns : {len(data_numerical_columns.columns)} columns")
            
            for column in schema_numerical_columns:
                if column not in data_numerical_columns:
                    logging.error(f"columns:{column}, not found in the data")
                    
            logging.info('Validation for numerical columns succesfull!')
        
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
    def validate_categorical_columns(self):
        
        logging.info("Validating categorical columns started ...")
        data_categorical_columns = list(self.raw_df.select_dtypes(include=['object']))
        schema_categorical_columns = list(self.schema['categorical_columns'])
        
        try:
        
            if len(data_categorical_columns) != len(schema_categorical_columns):
                logging.error(f"column mismatch occured, schema columns has: {len(schema_categorical_columns)} columns, data columns : {len(data_categorical_columns.columns)} columns")
            
            for column in schema_categorical_columns:
                if column not in data_categorical_columns:
                    logging.error(f"columns:{column}, not found in the data")
                    
            logging.info('Validation for categorical columns succesfull!')
        
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
    def run_validation(self):
        try:
            self.load_data()
            self.load_schema()
            self.validate_columns()
            self.validate_numerical_columns()
            self.validate_categorical_columns()
            
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
data_validation = DataValidation(artifact_raw_data_path, train_data_path, test_data_path, schema_file_path) 
data_validation.run_validation()
            
        
        
        
        