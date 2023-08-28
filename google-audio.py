from google.cloud import storage
import logging
import os
import cloudstorage as gcs
from google.cloud import speech

bucket_name = "mfd-audio"
source_file_name = "audio.flac"
destination_blob_name = "audio"

from pydub import AudioSegment
AudioSegment.converter = "C:\\Users\\Harishankar\\Downloads\\ffmpeg-4.3.1-win64-static\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\Users\\Harishankar\\Downloads\\ffmpeg-4.3.1-win64-static\\bin\\ffmpeg.exe"
AudioSegment.ffprobe ="C:\\Users\\Harishankar\\Downloads\\ffmpeg-4.3.1-win64-static\\bin\\ffprobe.exe"

from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS


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

app = Flask(__name__)
CORS(app)

@app.route('/audio', methods=['POST'])
def generate_summary():
    fileInput = request.files['audio']
    file = AudioSegment.from_file(fileInput,"flac")
    file.export("audio.flac",format = "flac")
    upload_blob()
    console.log("Uploaded")
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri="gs://mfd-audio/audio")
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=16000,
        language_code="en-US"
    )
    operation = client.long_running_recognize(
        request={"config": config, "audio": audio}
        )
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout = 300)
    text = ""
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        text = text +" ." + result.alternatives[0].transcript
    return jsonify(text)

    
if __name__ == '__main__':
    app.run(debug=True)