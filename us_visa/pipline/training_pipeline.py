import os
import sys 
from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *
from dataclasses import dataclass, field

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer
from us_visa.components.model_evaluation import ModelEvaluator
from us_visa.components.model_pusher import ModelPusher



data_ingestion = DataIngestion() 
data_ingestion.import_and_split()

data_transformation = DataTransformation(data_path,traindata_path, testdata_path, schema_file_path, trans_traindata_path, trans_testdata_path,preprocessor_path )
data_transformation.preprocess_traindata()

trainer = ModelTrainer(trans_traindata_path, model_path,random_state)
trainer.run_trainer()

evaluator = ModelEvaluator(model_path, testdata_path, preprocessor_path, schema_file_path)
evaluator.run_evaluation()

model_pusher = ModelPusher(model_path, bucket_name, s3_model_key, aws_access_key_id, aws_secret_access_key)
model_pusher.push_model()
    