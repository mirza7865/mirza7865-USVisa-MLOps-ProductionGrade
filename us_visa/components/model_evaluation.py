import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import yaml
import joblib  # Import joblib
import zipfile # Import zipfile

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *
from sklearn.metrics import accuracy_score, classification_report

@dataclass
class ModelEvaluator:
    model_path: str
    testdata_path: str
    preprocessor_path: str
    schema_file_path: str

    def load_model(self):
        try:
            logging.info(f"Loading model from {self.model_path}...")
            # Unzip the model first
            temp_pkl_path = "temp_model.pkl"
            with zipfile.ZipFile(self.model_path, 'r') as zipf:
                zipf.extract(os.path.basename(temp_pkl_path), path=os.path.dirname(self.model_path)) #extract to the same directory as the zip

            # Load the unzipped model
            model_full_path = os.path.join(os.path.dirname(self.model_path), temp_pkl_path)
            self.model = joblib.load(model_full_path) #Use joblib to load the model.

            os.remove(model_full_path) #Remove the temporary pkl file.

            logging.info("Model loaded and unzipped successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def load_preprocessor(self):
        try:
            logging.info(f"Loading preprocessor from {self.preprocessor_path}...")
            with open(self.preprocessor_path, "rb") as file:
                self.preprocessor = pickle.load(file)
            logging.info("Preprocessor loaded successfully.")
            # Debug: Print preprocessor transformers
            logging.info(f"Loaded preprocessor transformers: {self.preprocessor.named_transformers_}")
        except Exception as e:
            logging.error(f"Error loading preprocessor: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def load_test_data(self):
        try:
            logging.info("Loading raw test data...")
            self.test_df = pd.read_csv(self.testdata_path)
            logging.info("Raw test data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading test data: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def load_schema(self):
        try:
            logging.info("Reading schema file from config folder")
            with open(self.schema_file_path, 'r') as file:
                self.schema = yaml.safe_load(file)
            logging.info("schema file loaded succesfully!")

        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def extract_schema_columns(self):
        try:
            logging.info("Extracting dropping columns from schema file ...")
            self.drop_cols = self.schema.get('drop_columns', [])
            self.one_hot_cols = self.schema.get('oh_columns', [])
            self.num_features_cols = self.schema.get('num_features', [])
            self.ordinal_cols = self.schema.get('or_columns', [])
            self.transform_cols = self.schema.get('transform_columns', [])
            logging.info("Columns from schema file extracted succesfully!")

        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def preprocess_test_data(self):
        try:
            logging.info("Preprocessing test data...")

            # 1. Feature Engineering: company_age
            if "yr_of_estab" in self.test_df.columns:
                self.test_df["company_age"] = CURRENT_YEAR - self.test_df["yr_of_estab"]
            else:
                logging.warning("yr_of_estab column not found in test data.")
            print(self.test_df.columns)
            # 2. Column Dropping
            self.test_df.drop(self.drop_cols, axis=1, inplace=True)
            print(self.test_df.columns)
            # 3. Duplicate Removal
            self.test_df.drop_duplicates(inplace=True)

            # Separate features and target
            self.X_test = self.test_df.drop('case_status', axis=1)
            self.y_test = self.test_df['case_status']

            status_mapping = {"Denied": 0, "Certified": 1}
            self.y_test = self.y_test.map(status_mapping)

            # 4. Apply preprocessor (TRANSFORM, NOT FIT_TRANSFORM)
            test_transformed = self.preprocessor.transform(self.X_test)

            # 5. Get transformed column names
            one_hot_encoded_cols = list(self.preprocessor.named_transformers_['one_hot']['one_hot'].get_feature_names_out(self.one_hot_cols))
            transformed_columns = self.num_features_cols + self.ordinal_cols + one_hot_encoded_cols + [col for col in self.X_test.columns if col not in self.num_features_cols + self.ordinal_cols + self.one_hot_cols + self.transform_cols]

            # 6. Convert NumPy array to DataFrame
            self.X_test = pd.DataFrame(test_transformed, columns=transformed_columns)

            # Combine transformed features and encoded target
            test_transformed_df = pd.concat([self.X_test, self.y_test.reset_index(drop=True)], axis=1)

            test_transformed_df.to_csv('artifact/model_evaluation/trans_testdata.csv', index=False)

            logging.info("Test data preprocessing completed successfully.")

        except Exception as e:
            logging.error(f"Error preprocessing test data: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def evaluate_model(self):
        try:
            logging.info("Evaluating model...")
            # X_test = self.test_df.drop("case_status", axis=1)
            # y_test = self.test_df["case_status"]
            y_pred = self.model.predict(self.X_test )
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            logging.info(f"Test Accuracy: {accuracy}")
            logging.info(f"Classification Report:\n{report}")
            return accuracy, report
        except Exception as e:
            logging.error(f"Error evaluating model: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def run_evaluation(self):
        try:
            self.load_model()
            self.load_preprocessor()
            self.load_test_data()
            self.load_schema()
            self.extract_schema_columns()
            self.preprocess_test_data()
            self.evaluate_model()
        except Exception as e:
            logging.error(f"Evaluation pipeline failed: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

# Example usage (replace with your actual paths)

evaluator = ModelEvaluator(model_path, testdata_path, preprocessor_path, schema_file_path)
evaluator.run_evaluation()