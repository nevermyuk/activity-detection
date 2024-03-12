"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import glob
import os
import subprocess

import cv2

videodir = "D:\SIT\Year3-1\ICT3104/Project/nvda-ml-activity-detection/step/videos/"  # TODO: put the path to your AVA dataset here
root = os.path.dirname(videodir)
outdir_clips = os.path.join(root, "frame/")

clip_length = 1  # seconds
clip_time_padding = 1.0  # seconds


# utils
def hou_min_sec(millis):
    millis = int(millis)
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (60 * 1000)) % 60)
    hours = int((millis / (60 * 60 * 1000)) % 60)
    return "%d:%d:%d" % (hours, minutes, seconds)


videonames = glob.glob(videodir + "*")
videonames = [os.path.basename(v).split(".")[0] for v in videonames]

for video_id in videonames:
    videofile = glob.glob(os.path.join(videodir, video_id + "*"))[0]
    clips_dir = os.path.join(outdir_clips, video_id)
    if not os.path.isdir(clips_dir):
        os.makedirs(clips_dir)

    print("Working on", video_id)
    ffmpeg_command = f"ffmpeg -i {videofile} -start_number 0 -qscale:v 4 {os.path.join(clips_dir,'%06d.jpg')}"

    subprocess.call(ffmpeg_command, shell=True)
