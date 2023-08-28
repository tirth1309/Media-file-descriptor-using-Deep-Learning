import pandas as pd
import os
import cv2
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

required_video_file = "./Downloaded_test_videos/8MVo7fje_oE.avi"
output_video_path = "./Test_videos_2/"

#with open("times.txt") as f:
#  times = f.readlines()

#times = [x.strip() for x in times] 

starttime = 0
endtime = 15
for i in range(20):
  #starttime = int(time.split("-")[0])
  #endtime = int(time.split("-")[1])
  ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname= output_video_path + "gKHQ715IMyc_"+str(starttime)+"_"+str(endtime)+".avi")
  starttime = int(starttime)
  endtime = int(endtime)
  starttime = starttime + 10
  endtime = endtime + 10
  if endtime == 200:
  	break

  #starttime = str(starttime)
  #endtime = str(endtime)

