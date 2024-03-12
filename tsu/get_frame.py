import os
import time
from pathlib import Path

import cv2


def video_to_frames(input_loc, video_name, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video directory
        video_name: Selected video id
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(f"{input_loc}/{video_name}.mp4")
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")

    video_feat_output = f"{output_loc}/{video_name}"
    Path(video_feat_output).mkdir(parents=True, exist_ok=True)

    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(video_feat_output + "/%#08d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break


if __name__ == "__main__":
    input_loc = "../data/TSU/TSU_Videos_mp4"
    video_name = "P02T02C06"
    output_loc = "../data/TSU/TSU_Videos_mp4"
    video_to_frames(input_loc, video_name, output_loc)
