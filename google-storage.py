from google.cloud import storage
import logging
import os
import cloudstorage as gcs
#import webapp2

#from google.appengine.api import app_identity
bucket_name = "mfd-audio"
source_file_name = "peacock.wav"
destination_blob_name = "audio"


def upload_blob():
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
upload_blob()
# self.response.write('Reading the full file contents:\n')
#     gcs_file = gcs.open(filename)
#     contents = gcs_file.read()
#     gcs_file.close()
#     self.response.write(contents)