import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import joblib
import json
import zipfile
from us_visa.logger import logging  # Assuming you have a logger
from us_visa.exception import USvisaException  # Assuming you have an exception class
from us_visa.constants import preprocessor_path, model_path, schema_file_path, CURRENT_YEAR, status_mapping_path # Assuming you have constants
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_object(file_path):
    try:
        logging.info(f"Loading object from {file_path}")

        # Check if the file is a zip file
        if file_path.endswith('.zip'):
            # Unzip the model first
            temp_pkl_path = "temp_model.pkl"
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extract(os.path.basename(temp_pkl_path), path=os.path.dirname(file_path)) #extract to the same directory as the zip

            # Load the unzipped model
            model_full_path = os.path.join(os.path.dirname(file_path), temp_pkl_path)
            model = joblib.load(model_full_path) #use joblib to load the model.

            os.remove(model_full_path) #Remove the temporary pkl file.

            return model
        else:
            # Load the regular pickle/joblib file
            return joblib.load(file_path)

    except Exception as e:
        logging.error(f"Error loading object: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)

def load_schema(schema_file_path):
    try:
        logging.info(f"Loading schema from {schema_file_path}")
        with open(schema_file_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema
    except Exception as e:
        logging.error(f"Error loading schema: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)

def extract_schema_columns(schema):
    try:
        logging.info("Extracting schema columns")
        drop_cols = schema.get('drop_columns', [])
        one_hot_cols = schema.get('oh_columns', [])
        num_features_cols = schema.get('num_features', [])
        ordinal_cols = schema.get('or_columns', [])
        transform_cols = schema.get('transform_columns', [])
        return drop_cols, one_hot_cols, num_features_cols, ordinal_cols, transform_cols
    except Exception as e:
        logging.error(f"Error extracting schema columns: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)

def preprocess_input_data(input_df, preprocessor, schema_file_path):
    try:
        logging.info("Preprocessing input data")
        schema = load_schema(schema_file_path)
        drop_cols, one_hot_cols, num_features_cols, ordinal_cols, transform_cols = extract_schema_columns(schema)

        # 1. Feature Engineering: company_age
        if "yr_of_estab" in input_df.columns:
            input_df["company_age"] = CURRENT_YEAR - input_df["yr_of_estab"]
        else:
            logging.warning("yr_of_estab column not found in input data.")

        print(input_df.columns)
        # 2. Column Dropping
        input_df.drop(drop_cols, axis=1, inplace=True)
        print(input_df.columns)

        # 3. Duplicate Removal
        input_df.drop_duplicates(inplace=True)
        

        # 4. Apply preprocessor
        transformed_data = preprocessor.transform(input_df)

        # 5. Get transformed column names
        one_hot_encoded_cols = list(preprocessor.named_transformers_['one_hot']['one_hot'].get_feature_names_out(one_hot_cols))
        transformed_columns = num_features_cols + ordinal_cols + one_hot_encoded_cols + [col for col in input_df.columns if col not in num_features_cols + ordinal_cols + one_hot_cols + transform_cols]

        # 6. Convert NumPy array to DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
        print(transformed_df)
        return transformed_df
    
    except Exception as e:
        logging.error(f"Error preprocessing input data: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)

def predict(input_data):
    try:
        logging.info("Making predictions")
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        transformed_data = preprocess_input_data(input_data, preprocessor, schema_file_path)
        predictions = model.predict(transformed_data)
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)
    
def load_status_mapping(file_path):
    try:
        with open(file_path, "r") as file_obj:
            return json.load(file_obj)
    except FileNotFoundError:
        logging.error(f"Error: status mapping file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: invalid JSON in status mapping file at {file_path}")
        return None

status_mapping = load_status_mapping(status_mapping_path)
   
if __name__ == "__main__":
    # Example usage
    try:
        example_input = pd.DataFrame({
            "case_id": ['EZYV18348'],
            "continent": ['Asia'],
            "education_of_employee": ["Bachelor's"],
            "has_job_experience": ['N'],
            "requires_job_training": ['Y'],
            "no_of_employees": [14083],
            "yr_of_estab": [1914],
            "region_of_employment": ['West'],
            "prevailing_wage": [70048.73],
            "unit_of_wage": ['Year'],
            "full_time_position": ['Y']      

        })
        predictions = predict(example_input)
        inverse_mapping = {v: k for k, v in status_mapping.items()}
        readable_predictions = [inverse_mapping[pred] for pred in predictions]
        print("Predictions:",  readable_predictions)
    except Exception as e:
        print(f"Error in main: {e}")















