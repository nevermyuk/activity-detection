from __future__ import unicode_literals

import glob
import os
from pathlib import Path
from turtle import down

import yt_dlp


def my_hook(d):
    if d["status"] == "downloading":
        print("Downloading video!")
    if d["status"] == "finished":
        print("Downloaded!")


ydl_opts = {
    "outtmpl": os.path.join(os.getcwd() + "/videos", "%(id)s.%(ext)s"),
    "progress_hooks": [my_hook],
    "format": "mp4",
}


def process(line):
    print(line)


if __name__ == "__main__":
    downloaded = []
    for f in glob.glob("./videos/*.mp4"):
        downloaded.append(Path(f).stem)
    Path("./videos").mkdir(parents=True, exist_ok=True)
    video_ids = []
    # load in videos
    with open("./datasets/vid_exists.txt") as f:
        for line in f:
            video_ids.append(line.rstrip())
    for video in downloaded:
        video_ids.remove(video)
    with open("blocked.txt", "w") as f:
        print("This message will be written to a file.", file=f)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for id in video_ids:
                try:
                    ydl.download([f"https://www.youtube.com/watch?v={id}"])
                except:
                    print(id, file=f)
                    continue
