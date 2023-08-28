import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
#from google.cloud import speech

# Instantiates a client
client = speech.SpeechClient()

# The name of the audio file to transcribe
#file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\LibriSpeech\dev-clean\84\121123"),"121123", "84-121123-0001.flac")
file_name = os.path.join("peacock.wav")
#file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\MP\audio"),"audio", "OSR_us_000_0010_8k.wav")
#file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\MediaFileDescriptor"),"MediaFileDescriptor","trial.flac")

# Loads the audio into memory
#with io.open(file_name, "rb") as audio_file:
#    content = audio_file.read()
#    audio = speech.RecognitionAudio(content=content)
    
# gcp
audio = speech.RecognitionAudio(uri="gs://mfd-audio/audio")

'''config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
    sample_rate_hertz=16000,
    language_code="en-US"
)'''
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #sample_rate_hertz=16000,
    language_code="en-US"
)
  # [START speech_python_migration_async_respons
operation = client.long_running_recognize(
	request={"config": config, "audio": audio}
    )
operation = client.long_running_recognize(config=config, audio=audio)

# Detects speech in the audio file
response = operation.result(timeout = 90)
text = []
for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))
    print("Confidence: {}".format(result.alternatives[0].confidence))
    text.append(result.alternatives[0].transcript)
print(text)