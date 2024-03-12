# -*- coding: utf-8 -*-

# Import Libraries
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir=None):
    """
    Sets up S3 client with given credentials and downloads a specified folder from S3 to a local directory.

    Parameters:
    aws_access_key_id (str): AWS access key ID.
    aws_secret_access_key (str): AWS secret access key.
    bucket_name (str): Name of the S3 bucket.
    s3_folder (str): Folder in the S3 bucket to download.
    local_dir (str, optional): Local directory to download the files to. Defaults to the S3 folder name.
    """

    # Create an S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    except PartialCredentialsError:
        print("Your AWS credentials are incomplete.")
        return
    except NoCredentialsError:
        print("Your AWS credentials were not found.")
        return
    # Input of function is the bucket name (which is always deeplearningcvfood),
    # the S3 folder which can either be: test, train, validation
    # and a default local directory
    def download_s3_folder(bucket_name, s3_folder, local_dir=None):
        if local_dir is None:
            local_dir = s3_folder
        # AWS S3 paginators are used to retrieve the objects in the bucket in multiple pages
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            # Iterate over every object on current page
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
                for obj in page.get('Contents', []):
                    # Captures the key associate with each image (file)
                    file_key = obj['Key']
                    if file_key.endswith('/'):
                        continue  # skip directories
                    local_file_path = os.path.join(local_dir, os.path.basename(file_key))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # Actual downloading of S3 object to local file path
                    s3_client.download_file(bucket_name, file_key, local_file_path)
                    print(f"Downloaded {file_key} to {local_file_path}")
        except NoCredentialsError:
            print("Invalid AWS credentials")
        except s3_client.exceptions.NoSuchBucket:
            print("The bucket does not exist or you have no access.")
        except Exception as e:
            print(e)

    # Call the download function
    download_s3_folder(bucket_name, s3_folder, local_dir)

# Example Usage of function
# aws_access_key_id = 'AKIA6GBMB6KOLQDCIA7C'
# aws_secret_access_key = '0HGaGhEVSm8cHNYbFQttvtEEZIzdm5nJ0DDTz0Xx'
# bucket_name = 'drinkutensillaptopdata'
# # Please set the exact file path of the data folder in the S3 bucket. 
# # The Detectron2 path is as follows for images: 'Detectron2/images/test', 'Detectron2/images/train', 'Detectron2/images/validation'
# # The Detectron2 path is as follows for annotations: 'Detectron2/annotations/train_annotations.json', 'Detectron2/annotations/val_annotations.json', 'Detectron2/annotations/test_annotation.json'
# # The YOLO path is as follows for images: 'YOLO/images/train', 'YOLO/images/val', 'YOLO/images/test'
# # The YOLO path is as follows for annotations: 'YOLO/labels/train', 'YOLO/labels/val', 'YOLO/labels/test'
# s3_folder = 'Detectron2/images/test'
# local_dir = '/Users/sveerisetti/Desktop/Trial'
# setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir)