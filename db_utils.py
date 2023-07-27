import json
import pymongo
import datetime
from pymongo import MongoClient
import cv2
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError

SECRET_KEY = 'P7pclPRbKFyAc2uTM1vfQ15QQEE7fXBonEmYobyz'
BUCKET_NAME = 'bucket-big-basket'
ACCESS_KEY = 'AKIAVUIZU62PZRBFNQTA'
dbCollection = None

get_url = lambda s3_file_name: f'https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_file_name}'

def upload_to_aws(frame_path, bucket, s3_file_name):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(str(frame_path), bucket, s3_file_name)
        print("Upload Successful")
        return get_url(s3_file_name)
    
    except FileNotFoundError:
        print("The file was not found")
        return ''   

    except NoCredentialsError:
        print("Credentials not available")
        return ''

def upload_file_to_s3(file,  st, bucket_name = 'bucket-big-basket',acl="public-read"):
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
        Path_of_upload = "Vehicles/" + st
        print(Path_of_upload)

        s3.put_object(Bucket=bucket_name, Key = Path_of_upload, Body=file,ACL =acl)
        print("Upload Successful")
        return f'https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/Vehicles/{st}'
    except:
        return "Failed"

def get_database():
    # Provide the mongodb atlas url
    CONNECTION_STRING = "mongodb+srv://hb:0709@cluster0.mkorr.mongodb.net/coromandel?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    dbname=client['adityabirla']
    collection_name = dbname["All_Alerts"]
    return collection_name
        
def push_to_db(v_frame, db_collection):
    # db_collection = dbCollection
    now = datetime.now()
    time_1 = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    im_id = str(date)+str(time_1)
    img_encode = cv2.imencode('.jpg', v_frame)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tobytes()
    url = upload_file_to_s3(str_encode,im_id+'.jpg')
    print("url:", url)
    item= {
            "Camera": "CCU Warehouse",
            "Date": date,
            "Time": time_1,
            "Alert_Type": "Overspeeding",
            "Status": "Warning",
            "Image": url
            }
    dbCollection.insert_one(item)
    return True

dbCollection = get_database()
if dbCollection is None:
        print("Unable to connect to database")

