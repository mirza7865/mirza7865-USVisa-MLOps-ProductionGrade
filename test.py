from flask import Flask, request, jsonify
import os
import sys
import pandas as pd
import yaml
import joblib
import zipfile
import json
from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import preprocessor_path, model_path, schema_file_path, CURRENT_YEAR, status_mapping_path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_object(file_path):
    try:
        logging.info(f"Loading object from {file_path}")
        if file_path.endswith('.zip'):
            temp_pkl_path = "temp_model.pkl"
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extract(os.path.basename(temp_pkl_path), path=os.path.dirname(file_path))
            model_full_path = os.path.join(os.path.dirname(file_path), temp_pkl_path)
            model = joblib.load(model_full_path)
            os.remove(model_full_path)
            return model
        else:
            return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading object: {USvisaException(e, sys)}")
        raise USvisaException(e, sys)

preprocessor = load_object(preprocessor_path)
# print(preprocessor.named_transformers_['one_hot']['one_hot'].categories_)

ordinal_encoder = preprocessor.named_transformers_['ordinal']['ordinal']
print(ordinal_encoder.categories_)