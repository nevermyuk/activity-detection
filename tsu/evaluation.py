from __future__ import division

import argparse
import datetime
import os
import pickle
import random
import sys
import time
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

import wandb


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_file", type=str)
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument("-dataset", type=str, default="charades")
    parser.add_argument("-root", type=str, default="no_root")
    parser.add_argument("-model", type=str, default="")
    parser.add_argument("-APtype", type=str, default="wap")
    parser.add_argument("-randomseed", type=str, default="False")
    parser.add_argument("-load_model", type=str, default="False")
    parser.add_argument("-batch_size", type=str, default="False")
    parser.add_argument("-split_setting", type=str)
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_result_output(input, file_name):
    event_labels = load_labels()
    col_name = ["Video", "Validation mAP"]
    col_name.extend(event_labels)
    df = pd.DataFrame(input, columns=col_name)
    save_dir_path = "./results"
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{save_dir_path}/{file_name}.csv", header=col_name, index=False)


def load_labels():
    df = pd.read_csv("../data/TSU/labels.csv")
    df = df.iloc[:, 1:]
    return df["Event"].values.tolist()


def load_data(train_split, val_split, root, classes):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, "training", root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dataloader.root = root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, "testing", root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_dataloader.root = root

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}
    return dataloaders, datasets


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
    # create empty pandas DF with column name
    result_csv = []
    original_stdout = sys.stdout  # Save a reference to the original standard output
    time_now = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_name = f"./results/{time_now}.txt"
    with open(result_name, "w") as result_file:
        model.eval()
        results = {}
        apmCurrent = APMeter()
        apm = APMeter()
        tot_loss = 0.0
        error = 0.0
        num_iter = 0.0
        num_preds = 0
        full_probs = {}
        for count, data in enumerate(tqdm(dataloader, desc="Evaluating model..")):
            sleep(0.01)
            num_iter += 1
            other = data[3]
            with torch.no_grad():
                outputs, loss, probs, err = run_network(
                    model, data, 0, baseline, classes
                )
            fps = outputs.size()[1] / other[1][0]
            apmCurrent.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
            apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
            error += err.data
            tot_loss += loss.data
            probs = probs.squeeze()
            full_probs[other[0][0]] = probs.data.cpu().numpy().T
            val_map_current = (
                torch.sum(100 * apmCurrent.value())
                / torch.nonzero(100 * apmCurrent.value()).size()[0]
            )
            sys.stdout = (
                result_file  # Change the standard output to the file we created.
            )

            # Print to file
            print(f"Video Name: {other[0][0]}")
            print("Validation mAP:", val_map_current)
            print("The Average Precision Per Activity Class")
            print(100 * apmCurrent.value())
            print(f"The shape is ({outputs.size()[2]}, {outputs.size()[1]})")

            # Print to terminal
            sys.stdout = (
                original_stdout  # Reset the standard output to its original value
            )
            print(f"Video Name: {other[0][0]}")
            print("Validation mAP:", val_map_current)
            print("Average Precision Per Activity Class")
            print(100 * apmCurrent.value())
            print(f"The shape is ({outputs.size()[2]}, {outputs.size()[1]})")
            results[other[0][0]] = (
                outputs.data.cpu().numpy()[0],
                probs.data.cpu().numpy()[0],
                data[2].numpy()[0],
                fps,
            )
            # add to df
            current = [other[0][0], val_map_current.item()]
            result_csv.append(current + (100 * apmCurrent.value()).tolist())

            wandb.log(
                {
                    "Validation mAP": val_map_current,
                    "Validation Error": err.data,
                    "Validation Loss": loss.data,
                    "Validation Average Precision Per Activity": 100
                    * apmCurrent.value(),
                }
            )
            apmCurrent.reset()

        epoch_loss = tot_loss / num_iter
        mean_error = error / num_iter
        val_map = (
            torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
        )
        sys.stdout = result_file  # Change the standard output to the file we created.
        print("Mean Validation mAP:", val_map)
        print("Mean Average Precision Per Activity Class")
        print(100 * apm.value())
        print(
            f"The shape is ({len(full_probs.keys())}, {sum(len(v) for v in full_probs.values())})"
        )
        # Print to terminal
        sys.stdout = original_stdout  # Reset the standard output to its original value
        print("Mean Validation mAP:", val_map)
        print("Mean Average Precision Per Activity Class")
        print(100 * apm.value())
        print(
            f"The shape is ({len(full_probs.keys())}, {sum(len(v) for v in full_probs.values())})"
        )
        # Add to data
        mean = ["MEAN", val_map.item()]

        wandb.log(
            {
                "Mean Validation mAP": val_map,
                "Mean Validation Error": mean_error,
                "Mean Validation Loss": epoch_loss,
                "Mean Validation Average Precision Per Activity": 100 * apm.value(),
            }
        )
        result_csv.append(mean + (100 * apm.value()).tolist())
        apm.reset()
    # create dataframe
    create_result_output(result_csv, time_now)
    return results


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

    # Wandb
    wandb.config = {
        "dataset": args.dataset,
        "split_settings": args.split_setting,
        "model": args.model,
        "batch_size": args.batch_size,
        "load_model": args.load_model,
        "root": args.root,
    }

    wandb.init(
        project="nvda-ml-activity-detection",
        entity="ict3104-team14-2022",
        config=wandb.config,
        job_type="evaluation",
    )
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
        split_setting = str(args.split_setting)
        from smarthome_i3d_per_video import TSU as Dataset
        from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

    train_split = args.split_setting
    test_split = args.split_setting

    input_channel = 1024
    # Set classes, 51 for TSU.
    num_classes = 51

    root = str(args.root)

    # Printing
    print(str(args.model))
    print("batch_size:", batch_size)
    print("Cuda Available: ", torch.cuda.is_available())
    print("GPU with CUDA :", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    # Set Stream
    dataloaders, datasets = load_data(train_split, test_split, root, num_classes)

    if args.load_model != "False":
        print("Evaluating....")
        # Add more models here if required.
        if args.model == "PDAN":
            print("PDAN Model")
            import models

            model = models.PDAN
            if args.load_model != "False":
                model = torch.load(args.load_model)
                print("loaded", args.load_model)
            model.cuda()
            result = eval_model(model, dataloaders["val"], num_classes)
            print("We are done.")
        # weight
        # model.load_state_dict(torch.load(str(args.load_model)))
        else:
            print("Please select a valid CNN model (PDAN)")
    else:
        print("No model loaded. Please add a model path to -load_mdel")
