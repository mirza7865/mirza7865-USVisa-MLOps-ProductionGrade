import boto3
import os
import pickle
import sys
from dataclasses import dataclass
import joblib  # Import joblib
import zipfile # Import zipfile

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *


@dataclass
class ModelPusher:
    model_path: str
    bucket_name: str
    s3_model_key: str
    aws_access_key_id: str  # Add access key
    aws_secret_access_key: str  # Add secret access key

    def push_model(self):
        """
        Unzips the model and pushes the .pkl file to an S3 bucket.
        """
        try:
            # 1. Unzip the model
            temp_pkl_path = "temp_model.pkl"
            with zipfile.ZipFile(self.model_path, 'r') as zipf:
                zipf.extract(os.path.basename(temp_pkl_path), path=os.path.dirname(self.model_path))

            # 2. Upload the .pkl file to S3
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )

            logging.info(f"Uploading model from {temp_pkl_path} to s3://{self.bucket_name}/{self.s3_model_key}...")

            full_temp_pkl_path = os.path.join(os.path.dirname(self.model_path), temp_pkl_path)

            s3_client.upload_file(full_temp_pkl_path, self.bucket_name, self.s3_model_key)

            logging.info("Model uploaded successfully.")

            # 3. Cleanup: Remove the temporary .pkl file
            os.remove(full_temp_pkl_path)

        except Exception as e:
            logging.error(f"Error uploading model to S3: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

model_pusher = ModelPusher(model_path, bucket_name, s3_model_key, aws_access_key_id, aws_secret_access_key)
model_pusher.push_model()
