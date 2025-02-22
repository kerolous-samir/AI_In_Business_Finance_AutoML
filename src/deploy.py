import sagemaker
import boto3
import os
from sagemaker.amazon.amazon_estimator import get_image_uri

start_session = sagemaker.Session()

bucket = 'gina-ai'
prefix = 'XGBoost-classifier'
key = 'XGBoost-classifier'

role = sagemaker.get_execution_role()

with open('../train.csv', 'rb') as f:
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(f)

s3_train_data = f's3://{bucket}/{prefix}/train/{key}'
print(f"Uploaded training data location {s3_train_data}")

with open('../validation.csv', 'rb') as f:
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', key)).upload_fileobj(f)

s3_validation_data = f's3://{bucket}/{prefix}/validation/{key}'
print(f"Uploaded validation data location {s3_validation_data}")

output_location = f's3://{bucket}/{prefix}/output'
print(f"Training artifacts will be uploaded to: {output_location}")

container = get_image_uri(boto3.Session().region_name, 'xgboost', 'latest')

xgboost_classifier = sagemaker.estimator.Estimator(container, role,
                                                   train_instance_count=1,
                                                   train_instance_type="ml.m4.xlarge",
                                                   output_path=output_location,
                                                   sagemaker_session=start_session)

xgboost_classifier.set_hyperparameters(max_depth=3,
                                       objective="multi:softmax",
                                       num_class=2,
                                       eta=0.5,
                                       num_round=150)

data_channels = {'train': s3_train_data, 'validation': s3_validation_data}
xgboost_classifier.fit(data_channels)

xgboost_classifier = xgboost_classifier.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

print("Model deployed successfully!")
