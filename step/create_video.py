import argparse
import glob
import os
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-selected_video", type=str, default="False")
    parser.add_argument("-path", type=str, default=f"{Path().absolute().as_posix()}")
    args = parser.parse_args()
    return args


def set_video_save_location(file_name):
    ## saving file
    save_dir_path = "./inference_video"
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)
    save_file_name = f"{save_dir_path}/inference_{file_name}.mp4"
    print(f"Saved at {save_file_name}")
    return save_file_name


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if str(args.selected_video):
        selected_file = str(args.selected_video)
        path_input = str(args.path)
        save_file_name = set_video_save_location(selected_file)
        selected_path = f"{path_input}/datasets/demo/frames/results/{selected_file}"
        os.listdir(selected_path)
        print(selected_path)
        fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
        out = cv2.VideoWriter(save_file_name, fourcc, 25.0, (640, 480))
        for img in sorted(os.listdir(selected_path)):
            img_path = selected_path + "/" + img
            img = cv2.imread(img_path)
            out.write(img)
        out.release()

    else:
        print("Select a video.")
