from __future__ import division

import argparse
import random
import sys
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm.autonotebook import tqdm


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument("-dataset", type=str, default="charades")
    parser.add_argument("-root", type=str, default="no_root")
    parser.add_argument("-model", type=str, default="")
    parser.add_argument("-APtype", type=str, default="wap")
    parser.add_argument("-randomseed", type=str, default="False")
    parser.add_argument("-load_model", type=str, default="False")
    parser.add_argument("-batch_size", type=str, default="False")
    parser.add_argument("-video_name", type=str, default="False")
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_labels():
    df = pd.read_csv("../data/TSU/labels.csv")
    df = df.iloc[:, 1:]
    return df["Event"].tolist()


def create_prediction_output(input, file_name):
    df = pd.DataFrame(input)
    col_name = ["event", "start_frame", "end_frame"]
    save_dir_path = "./inference_prediction"
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{save_dir_path}/{file_name}.csv", header=col_name, index=False)


def load_data(root, video_name, classes):
    # Load Data
    dataset = Dataset(root, video_name, batch_size, classes)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dataloader.root = root
    return dataloader, dataset


def getSeed(randomseed):
    # set random seed
    if randomseed == "False":
        seed = 0
    elif randomseed == "True":
        seed = random.randint(1, 100000)
    else:
        seed = int(randomseed)
    return seed


def eval_model(model, dataloader, baseline=False, classes=51):
    model.eval()
    event_labels = load_labels()
    results = {}
    for count, data in enumerate(tqdm(dataloader, desc="Classifying actions..")):
        sleep(0.01)
        other = data[3]
        with torch.no_grad():
            outputs, loss, probs, err = run_network(model, data, 0, baseline, classes)
        frame_activation = torch.round(probs).data.cpu().numpy()
        frame_activation_reshaped = frame_activation.reshape(
            -1, frame_activation.shape[-1]
        )
        fps = outputs.size()[1] / other[1][0]
        # predicted = np.argmax(probs.data.cpu().numpy()[0], axis=1)
        number_of_frames = 1 / fps.numpy()
        current_frame = 0
        all_event = []
        for activations in frame_activation_reshaped:
            indexes = np.where(activations == 1.0)
            start_frame = current_frame
            end_frame = start_frame + number_of_frames
            current_frame = end_frame
            for events in indexes:
                for event in events:
                    current_event = event_labels[event]
                    all_event.append([current_event, start_frame, end_frame])
        create_prediction_output(all_event, other[0][0])
    print(f"The prediction is saved at ./inference_prediction/{other[0][0]}.csv")


def run_network(model, data, gpu, epoch=0, baseline=False, classes=51):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, : int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)
    outputs_final = activation
    if args.model == "PDAN":
        # print('outputs_final1', outputs_final.size())
        outputs_final = outputs_final[:, 0, :, :]

    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = torch.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, reduction="none")
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)
    return outputs_final, loss, probs_f, corr / tot


if __name__ == "__main__":

    # Parsing Argument
    args = parse_args()

    # Set Parameters
    if str(args.APtype) == "map":
        from apmeter import APMeter

    batch_size = int(args.batch_size)
    # CUDA stuff
    # Get Seed
    SEED = getSeed(args.randomseed)

    # Set Torch parameters
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ########## MODEL ###########
    if args.dataset == "TSU":
        from smarthome_i3d_per_video_inference import TSU as Dataset
        from smarthome_i3d_per_video_inference import TSU_collate_fn as collate_fn

    root = str(args.root)
    input_channel = 1024
    # Set classes, 51 for TSU.
    num_classes = 51

    video_name = str(args.video_name)

    # Printing
    print("Cuda Available: ", torch.cuda.is_available())
    print("GPU with CUDA :", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))

    if args.load_model != "False":
        print("Evaluating....")
        # Add more models here if required.
        if video_name != "False":
            dataloader, datasets = load_data(root, video_name, num_classes)

            if args.model == "PDAN":
                print("PDAN Model")
                import models

                model = models.PDAN
                model = torch.load(args.load_model)
                model.cuda()
                print("loaded", args.load_model)
                print(f"Inferencing {video_name}")
                result = eval_model(model, dataloader, num_classes)

            else:
                print("Only PDAN models are accepted... -args.models=PDAN")
        else:
            print("Please load valid video..")
        # weight
        # model.load_state_dict(torch.load(str(args.load_model)))
    else:
        print("No model loaded. Please add a model path to -load_mdel")
