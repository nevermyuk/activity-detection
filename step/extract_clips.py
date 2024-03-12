"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import glob
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-selected_video", type=str, default="False")
    args = parser.parse_args()
    return args


# utils
def hou_min_sec(millis):
    millis = int(millis)
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (60 * 1000)) % 60)
    hours = int((millis / (60 * 60 * 1000)) % 60)
    return "%d:%d:%d" % (hours, minutes, seconds)


def set_frame_save_location(video_id):
    ## saving file
    save_dir_path = f"./datasets/demo/frames/{video_id}"
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)
    return save_dir_path


if __name__ == "__main__":

    clip_length = 1  # seconds
    clip_time_padding = 1.0  # seconds

    # Parsing Argument
    args = parse_args()
    video_path = args.selected_video
    video_id = os.path.basename(video_path)
    clip_dir = set_frame_save_location(Path(video_id).stem)

    # videonames = glob.glob(videodir + "*")
    # videonames = [os.path.basename(v).split(".")[0] for v in videonames]

    # for video_id in videonames:
    # videofile = glob.glob(os.path.join(videodir, video_id + "*"))[0]
    # clips_dir = os.path.join(outdir_clips, video_id)
    # if not os.path.isdir(clips_dir):
    #     os.makedirs(clips_dir)
    print("Working on", video_path)
    ffmpeg_command = f"ffmpeg -i {video_path} -start_number 0 -qscale:v 4 {os.path.join(clip_dir,'%06d.jpg')}"

    subprocess.call(ffmpeg_command, shell=True)
