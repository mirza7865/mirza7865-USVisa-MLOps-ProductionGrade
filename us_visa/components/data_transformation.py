import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import yaml

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE # Import SMOTE


CURRENT_YEAR = date.today().year
trans_traindata_path= 'artifact/data_transformation/trans_train_data.csv'
trans_testdata_path= 'artifact/data_transformation/trans_test_data.csv'
preprocessor_path = 'artifact/data_transformation/preprocessor.pkl'

@dataclass
class DataTransformation:
    
    data_path:str
    traindata_path:str
    testdata_path:str
    schema_file_path:str
    trans_traindata_path:str
    trans_testdata_path:str
    preprocessor_path:str
    
    logging.info("Data tranformation component has started ...")
    
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
        
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
    def extract_schema_columns(self):
        try:
            logging.info("Extracting columns from schema file ...")
            self.drop_cols = self.schema.get('drop_columns', [])
            self.num_features_cols = self.schema.get('num_features', [])
            self.ordinal_cols = self.schema.get('or_columns', [])
            self.one_hot_cols = self.schema.get('oh_columns', [])
            self.transform_cols = self.schema.get('transform_columns', [])
            logging.info("Columns from schema file extracted succesfully!")
            
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
    
    def preprocess_traindata(self):
        try:
            self.load_data()
            self.load_schema()
            self.extract_schema_columns()

            logging.info("Preprocessing train_df and test_df started ...")

            logging.info("Feature engineering, adding company_age col for tran_df and test_df ...")
            self.train_df['company_age'] = CURRENT_YEAR - self.train_df['yr_of_estab']

            logging.info("removing unwanted columns ...")
            self.train_df.drop(self.drop_cols, axis=1, inplace=True)

            logging.info("Removing duplicates from train_df and test_df ...")
            self.train_df.drop_duplicates(inplace=True)

            # Numerical Pipeline
            logging.info("Imputing and scaling numerical columns...")
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Or 'mean'
                ('scaler', StandardScaler())
            ])

            logging.info("Power Transforming numerical columns...")
            transform_pipeline = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            # Ordinal Encoding Pipeline
            logging.info("Imputing and ordinal encoding ordinal columns...")
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder())
            ])

            # One-Hot Encoding Pipeline
            logging.info("Imputing and one hot encoding one hot columns...")
            one_hot_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info("Preprocessing pipeline started ...")
            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, self.num_features_cols),
                ('ordinal', ordinal_pipeline, self.ordinal_cols),
                ('one_hot', one_hot_pipeline, self.one_hot_cols)
            ], remainder='passthrough')
            
            # Fit and transform train data
            train_transformed = preprocessor.fit_transform(self.train_df)

            # Save the preprocessor
            with open(self.preprocessor_path, 'wb') as file:
                pickle.dump(preprocessor, file)
            logging.info(f"Preprocessor pickle file saved to {self.preprocessor_path}")

            one_hot_encoded_cols = list(preprocessor.named_transformers_['one_hot']['one_hot'].get_feature_names_out(self.one_hot_cols))
            transformed_columns = self.num_features_cols + self.ordinal_cols + one_hot_encoded_cols + [col for col in self.train_df.columns if col not in self.num_features_cols + self.ordinal_cols + self.one_hot_cols + self.transform_cols]

            # Debug: Print transformed data shape and column names
            logging.info(f"Shape of train_transformed: {train_transformed.shape}")
            logging.info(f"Length of transformed_columns: {len(transformed_columns)}")
            logging.info(f"First few elements of transformed_columns: {transformed_columns[:5]}")

            # Convert numpy array to dataframe.
            train_transformed_df = pd.DataFrame(train_transformed, columns=transformed_columns)

            logging.info("Preprocessing train and test pipeline Ended!")

            train_transformed_df.drop('case_status_Certified', axis=1, inplace=True)
            train_transformed_df = train_transformed_df.rename(columns={'case_status_Denied': 'case_status'})

            logging.info("Applying smote to deal with Imbalanced dataset ...")
            X_train = train_transformed_df.drop('case_status', axis=1)
            y_train = train_transformed_df['case_status']

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            train_transformed_df = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), y_train_resampled], axis=1)
            train_transformed = train_transformed_df.copy()

            logging.info("Exporting train_transformed_df and test_transformed_df to artifact ...")
            train_transformed_df.to_csv(self.trans_traindata_path, index=False)
            logging.info("Preprocessing completed!")

        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)
        
        
# data_transformation = DataTransformation(data_path,traindata_path, testdata_path, schema_file_path, trans_traindata_path, trans_testdata_path,preprocessor_path )
# data_transformation.preprocess_traindata()
