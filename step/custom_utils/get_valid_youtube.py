import os
from re import T

import pandas as pd
import requests


def check_video_url(video_id):
    checker_url = "https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v="
    video_url = checker_url + video_id

    request = requests.get(video_url)

    return request.status_code == 200


def load_ava(step_wd):
    df_val = pd.read_csv(
        f"{step_wd}/datasets/ava_val_v2.1.csv",
        header=None,
        usecols=[0],
        names=["videoid"],
    )
    df_train = pd.read_csv(
        f"{step_wd}/datasets/ava_train_v2.1.csv",
        header=None,
        usecols=[0],
        names=["videoid"],
    )

    val_unique = df_val["videoid"].unique().tolist()
    train_unique = df_train["videoid"].unique().tolist()
    videoids = val_unique + train_unique
    return videoids


def load_csv(step_wd, type):
    df_val = pd.read_csv(
        f"{step_wd}/datasets/ava_{type}_v2.1.csv",
        dtype=str,
        header=None,
        names=[
            "video_id",
            "middle_frame_timestamp",
            "x1",
            "y1",
            "x2",
            "y2",
            "action_id",
            "person_id",
        ],
    )
    return df_val


def save_csv(step_wd, type, df):
    df.to_csv(
        f"{step_wd}/datasets/ava_{type}_v2.1_filter.csv", header=False, index=False
    )


if __name__ == "__main__":
    step_wd = os.getcwd()
    val_df = load_csv(step_wd, "val")
    train_df = load_csv(step_wd, "train")

    video_ids = load_ava(step_wd)
    exist_dict = {}
    # check if video exists
    count = 0
    ######### Comment out this block after downloading######
    for id in video_ids:
        count += 1
        exist = check_video_url(id)
        if exist:
            print(id)
            exist_dict[id] = ""
    videos = []
    ########################################################
    ###########Uncomment after downloading videos ######
    # with open("./datasets/vid_exists.txt") as f:
    #     for line in f:
    #         videos.append(line.rstrip())
    # not_avail_videos = []
    # with open("./datasets/blocked.txt") as f:
    #     for line in f:
    #         not_avail_videos.append(line.rstrip())
    # for vid in not_avail_videos:
    #     try:
    #         videos.remove(vid)
    #     except:
    #         continue
    #######################################################
    for id in videos:
        exist_dict[id] = ""

    # drop if not in exist_dict
    val_df_stripped = val_df.drop(val_df[~val_df.video_id.isin(exist_dict)].index)
    train_df_stripped = train_df.drop(
        train_df[~train_df.video_id.isin(exist_dict)].index
    )

    # # reset the index
    val_df_stripped.reset_index(drop=True, inplace=True)
    train_df_stripped.reset_index(drop=True, inplace=True)
    # # save to csv
    save_csv(step_wd, "val", val_df_stripped)
    save_csv(step_wd, "train", train_df_stripped)
