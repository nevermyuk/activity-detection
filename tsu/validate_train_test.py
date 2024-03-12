import argparse
import json
import os
from tkinter import E


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-split_setting", type=str)
    parser.add_argument("-feature_dir", type=str)
    parser.add_argument("-output_path", type=str)
    args = parser.parse_args()
    return args


def load_split(split_file):
    # load the split file
    data = []
    try:
        with open(split_file, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print("Please select a valid split file.")
        print(e)


def load_features(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            # check if all files are numpy
            list_of_files = os.listdir(path)

            list_files_stripped = []
            for file in list_of_files:
                if not file.endswith(".npy"):
                    raise Exception("Please ensure directory only has .npy files.")
                else:
                    file_name, ext = os.path.splitext(file)
                    list_files_stripped.append(file_name)
            # return a list of all
            return list_files_stripped
        else:
            raise Exception("Please select a directory with only .npy files.")
    except Exception as e:
        print(e)


def create_json(split, features, output_file):
    try:
        dataset = {}
        for vid in features:
            try:
                video_name = vid.split("_", 1)[0]
                if video_name not in split.keys():
                    raise Exception(
                        f"{vid} either do not follow naming convention or do not exist in TSU dataset. Skipped."
                    )
                dataset[vid] = split[video_name]
            except Exception as e:
                print(e)
                continue

        with open(output_file, "w") as outfile:
            json.dump(dataset, outfile)
            print(f"Saved to: {output_file}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    args = parse_args()
    split = load_split(args.split_setting)
    features = load_features(args.feature_dir)
    create_json(split, features, args.output_path)
