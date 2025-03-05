import os
from datetime import date


####### Data Ingestion Component ########

raw_data_path = 'raw_data/EasyVisa.csv'
test_size=0.2
random_state=42
data_path = 'artifact/data_ingestion/raw_data.csv'
traindata_path= 'artifact/data_ingestion/train_data.csv'
testdata_path= 'artifact/data_ingestion/test_data.csv'

######## Data Validation Component ########

# data_path = 'artifact/data_ingestion/raw_data.csv'
# traindata_path= 'artifact/data_ingestion/train_data.csv'
# testdata_path= 'artifact/data_ingestion/test_data.csv'
schema_file_path = 'config/schema.yaml'

######## Data Transformation ######

CURRENT_YEAR = date.today().year
trans_traindata_path= 'artifact/data_transformation/trans_train_data.csv'
trans_testdata_path= 'artifact/data_transformation/trans_test_data.csv'
preprocessor_path = 'artifact/data_transformation/preprocessor.pkl'
status_mapping_path = 'artifact/data_transformation/status_mapping.json'


########## Data Trainer ######
model_path = 'artifact/model_training/model.pkl.zip'
random_state = 42
model_choice = 'random_forest'


######### Model Pusher #######
bucket_name = 'us-visa-bucket786'
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") # Get from environment variable
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

REGION_NAME = 'us-east-1'
s3_model_key = 'model.pkl'



ECR = '509399625983.dkr.ecr.eu-north-1.amazonaws.com/visa'

# (base) mirza@Mirzas-MacBook-Air ~ % export AWS_ACCESS_KEY_ID='AKIAXNGUVOD7XNBL3VMJ'
# (base) mirza@Mirzas-MacBook-Air ~ % export AWS_SECRET_ACCESS_KEY='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'