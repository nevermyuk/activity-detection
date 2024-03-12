import argparse
from fileinput import filename
from pathlib import Path
from time import sleep

import cv2
import pandas as pd
from tqdm.autonotebook import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-selected_video", type=str, default="False")
    parser.add_argument("-pop_up", type=str, default="True")
    args = parser.parse_args()
    return args


def load_prediction(file_name):
    df = pd.read_csv(f"./inference_prediction/{file_name}.csv")
    event = df["event"].tolist()
    start_frame = df["start_frame"].tolist()
    end_frame = df["end_frame"].tolist()
    return event, start_frame, end_frame


def load_ground_truth(file_name):
    df = pd.read_csv(f"../data/TSU/TSU_Annotations/{file_name}.csv")
    event = df["event"].tolist()
    start_frame = df["start_frame"].tolist()
    end_frame = df["end_frame"].tolist()
    return event, start_frame, end_frame


def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_TRIPLEX
    scale = 0.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 10
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    bg_pos_x = pos[0] - 4
    bg_pos_y = pos[1] + 4
    end_x = bg_pos_x + txt_size[0][0] + margin
    end_y = bg_pos_y - txt_size[0][1] - margin + 4

    cv2.rectangle(img, (bg_pos_x, bg_pos_y), (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def set_video_save_location(file_name):
    ## saving file
    save_dir_path = "./inference_video"
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)
    save_file_name = f"{save_dir_path}/inference_{file_name}.mp4"
    return save_file_name


if __name__ == "__main__":
    # Parsing Argument
    args = parse_args()

    # background color
    BG_WHITE = (255, 255, 255)
    BG_GREEN = (8, 255, 8)
    PREDICT_START_POS = 10
    GT_START_POS = 330
    if str(args.selected_video):
        selected_file = str(args.selected_video)
        popup = str(args.pop_up)
        popup_message = (
            "Real time video playback will be shown"
            if popup == "True"
            else "No pop-up video playback."
        )
        print(popup_message)
        save_file_name = set_video_save_location(selected_file)
        selected_path = f"../data/TSU/TSU_Videos_mp4/{selected_file}.mp4"
        # load video

        cap = cv2.VideoCapture(selected_path)
        # store number of correct predictions
        correct_prediction_frame_count = 0
        # Check if video opened successfully
        if cap.isOpened():
            # set codec
            fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
            out = cv2.VideoWriter(save_file_name, fourcc, 25.0, (640, 480))

            ## total frame count for tqdm
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tqdm(total=frame_count, desc="Adding captions..") as pbar:
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    # load prediction
                    (
                        prediction_event,
                        prediction_start_frame,
                        prediction_end_frame,
                    ) = load_prediction(selected_file)
                    gt_event, gt_start_frame, gt_end_frame = load_ground_truth(
                        selected_file
                    )
                    if ret == True:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        # increment progress bar
                        if current_frame % 100 == 0:
                            pbar.update(100)

                        __draw_label(frame, "Frame", (120, 330), BG_WHITE)
                        __draw_label(
                            frame, "Predicted", (PREDICT_START_POS, 330), BG_WHITE
                        )
                        __draw_label(frame, "Accuracy", (200, 330), BG_WHITE)
                        __draw_label(
                            frame, f"{int(current_frame)}", (120, 350), BG_WHITE
                        )
                        __draw_label(
                            frame, "Prediction", (PREDICT_START_POS, 390), BG_WHITE
                        )
                        __draw_label(
                            frame, "Ground Truth", (GT_START_POS, 390), BG_WHITE
                        )

                        gt_event_text = []
                        for count, value in enumerate(gt_event):
                            if (
                                current_frame >= gt_start_frame[count]
                                and current_frame <= gt_end_frame[count]
                            ):
                                gt_event_text.append(gt_event[count])

                        predicted_event_text = []
                        for count, value in enumerate(prediction_event):
                            if (
                                current_frame >= prediction_start_frame[count]
                                and current_frame <= prediction_end_frame[count]
                            ):
                                predicted_event_text.append(prediction_event[count])

                        predicted_text_starting_pos = 410
                        gt_text_starting_pos = 410
                        predicted_event_set = set(predicted_event_text)
                        gt_event_set = set(gt_event_text)
                        intersect = set(predicted_event_text) & set(gt_event_text)

                        # if there is intersection
                        if intersect:
                            correct_prediction_frame_count += 1
                            for event in intersect:
                                BG_COLOR = BG_GREEN
                                __draw_label(
                                    frame,
                                    event,
                                    (PREDICT_START_POS, predicted_text_starting_pos),
                                    BG_COLOR,
                                )
                                predicted_text_starting_pos += 20
                                __draw_label(
                                    frame,
                                    event,
                                    (GT_START_POS, gt_text_starting_pos),
                                    BG_COLOR,
                                )
                                gt_text_starting_pos += 20
                                # predicted frames
                                __draw_label(
                                    frame,
                                    f"{correct_prediction_frame_count}",
                                    (PREDICT_START_POS, 350),
                                    BG_COLOR,
                                )
                                accuracy = (
                                    correct_prediction_frame_count / current_frame * 100
                                )
                                __draw_label(
                                    frame,
                                    f"{accuracy:.1f}%",
                                    (200, 350),
                                    BG_COLOR,
                                )
                        # no intersect
                        else:
                            BG_COLOR = BG_WHITE
                            for event in predicted_event_set.difference(gt_event_set):
                                __draw_label(
                                    frame,
                                    event,
                                    (PREDICT_START_POS, predicted_text_starting_pos),
                                    BG_COLOR,
                                )
                                predicted_text_starting_pos += 20
                            for event in gt_event_set.difference(predicted_event_set):
                                __draw_label(
                                    frame,
                                    event,
                                    (GT_START_POS, gt_text_starting_pos),
                                    BG_COLOR,
                                )
                            if not len(gt_event_set) and not len(predicted_event_set):
                                correct_prediction_frame_count += 1
                                BG_COLOR = BG_GREEN

                            gt_text_starting_pos += 20
                            __draw_label(
                                frame,
                                f"{correct_prediction_frame_count}",
                                (PREDICT_START_POS, 350),
                                BG_COLOR,
                            )
                            # if gt is empty

                            # accuracy
                            accuracy = (
                                correct_prediction_frame_count / current_frame * 100
                            )

                            __draw_label(
                                frame,
                                f"{accuracy:.1f}%",
                                (200, 350),
                                BG_COLOR,
                            )

                        # write to file
                        out.write(frame)
                        # Display the resulting frame, Commented out so wait for video to be done...
                        if popup == "True":
                            cv2.imshow("Frame", frame)

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print(
                            "Hmm....the captioning was exited prematurely, please try again!"
                        )
                        break

                    if frame is None:
                        print("Captioning has completed.")
                        print("We are done!")
                        break

                # When everything done, release the video capture object
                pbar.update(100)
                cap.release()
                out.release()
                # Closes all the frames
                cv2.destroyAllWindows()
        else:
            print("Error opening video stream or file")

    else:
        print("Please select a valid video.")
