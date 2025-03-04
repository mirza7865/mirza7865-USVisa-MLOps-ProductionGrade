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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV


@dataclass
class ModelTrainer:
    trans_traindata_path: str
    model_path: str
    random_state:int

    def load_data(self):
        try:
            logging.info("Reading transformed data from artifact folder")
            self.train_df = pd.read_csv(self.trans_traindata_path)
            logging.info("loaded the data succesfully!")
        except Exception as e:
            logging.error(f"An Exception has occured : {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def train_best_model(self):
        try:
            logging.info("Training best model...")
            best_model = None
            best_score = -1
            models = {
                "random_forest": RandomForestClassifier(random_state=random_state),
                "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
                "logistic_regression": LogisticRegression(random_state=random_state, solver='liblinear')
            }
            params = {
                "random_forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
                "gradient_boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]},
                "logistic_regression": {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
            }
            X_train = self.train_df.drop("case_status", axis=1)
            y_train = self.train_df["case_status"]

            for model_name, model in models.items():
                grid_search = GridSearchCV(model, params[model_name], cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_

            self.model = best_model
            logging.info(f"Best model training completed. Best score: {best_score}")
        except Exception as e:
            logging.error(f"Error training best model: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    # def save_model(self):
    #     try:
    #         logging.info(f"Saving best trained model to {self.model_path}...")
    #         with open(self.model_path, "wb") as file:
    #             pickle.dump(self.model, file)
    #         logging.info("Model saved successfully.")
    #     except Exception as e:
    #         logging.error(f"Error saving model: {USvisaException(e, sys)}")
    #         raise USvisaException(e, sys)
    
    def save_model_as_zip(self):
        try:
            logging.info(f"Zipping and saving model to {self.model_path}...")

            temp_pkl_path = "temp_model.pkl"
            joblib.dump(self.model, temp_pkl_path)

            with zipfile.ZipFile(self.model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(temp_pkl_path, os.path.basename(temp_pkl_path))

            os.remove(temp_pkl_path)
            logging.info("Model zipped and saved successfully.")

        except Exception as e:
            logging.error(f"Error zipping model: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

    def run_trainer(self):
        try:
            self.load_data()
            self.train_best_model()
            self.save_model_as_zip()
            logging.info("Training pipeline completed.")
        except Exception as e:
            logging.error(f"Training pipeline failed: {USvisaException(e, sys)}")
            raise USvisaException(e, sys)

# trainer = ModelTrainer(trans_traindata_path, model_path,random_state)
# trainer.run_trainer()