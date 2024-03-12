from __future__ import division

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
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
    parser.add_argument("-comp_info", type=str)
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument("-dataset", type=str, default="charades")
    parser.add_argument("-root", type=str, default="no_root")
    parser.add_argument("-lr", type=str, default="0.1")
    parser.add_argument("-epoch", type=str, default="50")
    parser.add_argument("-model", type=str, default="")
    parser.add_argument("-APtype", type=str, default="wap")
    parser.add_argument("-randomseed", type=str, default="False")
    parser.add_argument("-load_model", type=str, default="False")
    parser.add_argument("-num_channel", type=str, default="False")
    parser.add_argument("-batch_size", type=str, default="False")
    parser.add_argument("-kernelsize", type=str, default="False")
    parser.add_argument("-split_setting", type=str)
    parser.add_argument("-trained_model_name", type=str, default="choose_a_model_name")
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        batch_size=batch_size,
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


# train the model
def train_model(models, criterion, num_epochs=50, classes=51):
    since = time.time()
    prev_best_model_path = None
    # prev_best_model_state_path = None
    best_map = 0.0
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            (
                train_map,
                train_loss,
                train_mean_average_precision_per_activity,
            ) = train_step(model, gpu, optimizer, dataloader["train"], epoch, classes)
            (
                prob_val,
                val_loss,
                val_map,
                val_mean_average_precision_per_activity,
            ) = val_step(model, gpu, dataloader["val"], classes)
            probs.append(prob_val)
            sched.step(val_loss)
            if best_map < val_map:
                print(
                    f"Curent epoch model has better Validation mAP: {val_map} than current best Validation mAP: {best_map} "
                )
                best_map = val_map
                save_dir_path = f"./model/trained/"
                Path(save_dir_path).mkdir(parents=True, exist_ok=True)
                # model_state_dict_path = f"{save_dir_path}/model_state_dict_{args.trained_model_name}_epoch_{epoch}"
                model_path = (
                    f"{save_dir_path}/model_{args.trained_model_name}_epoch_{epoch}"
                )
                # torch.save(
                #     model.state_dict(),
                #     model_state_dict_path,
                # )
                torch.save(model, model_path)
                print(f"New Model saved at {model_path}")

                if prev_best_model_path:
                    print(f"Removing previous model {prev_best_model_path} ")
                    os.remove(prev_best_model_path)

                prev_best_model_path = model_path
                # if prev_best_model_state_path:
                #     print(f"Removing previous model {prev_best_model_state_path} ")
                #     os.remove(prev_best_model_state_path)

                # prev_best_model_state_path = model_state_dict_path
        wandb.log(
            {
                "epoch": epoch,
                "Train mAP": train_map,
                "Train loss": train_loss,
                "Train Mean Average Precision Per Activity": train_mean_average_precision_per_activity,
                "Valid mAP": val_map,
                "Valid loss": val_loss,
                "Valid Mean Average Precision Per Activity": val_mean_average_precision_per_activity,
            }
        )


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


def train_step(model, gpu, optimizer, dataloader, epoch, classes):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.0
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch, classes)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()
    if args.APtype == "wap":
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print("Train mAP:", train_map)
    print("The Mean Average Precision Per Activity Class")
    print(100 * apm.value())

    train_mean_average_precision_per_activity = 100 * apm.value()
    apm.reset()

    epoch_loss = tot_loss / num_iter
    return train_map, epoch_loss, train_mean_average_precision_per_activity


def val_step(model, gpu, dataloader, classes):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.0
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, classes)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()
        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter
    val_mean_average_precision_per_activity = 100 * apm.value()

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print("Validation mAP:", val_map)
    print("The Mean Average Precision Per Activity Class")
    print(100 * apm.value())
    print(
        f"shape is ({len(full_probs.keys())}, {sum(len(v) for v in full_probs.values())})"
    )
    apm.reset()

    return full_probs, epoch_loss, val_map, val_mean_average_precision_per_activity


if __name__ == "__main__":

    # Parsing Argument
    args = parse_args()

    # Wandb
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epoch,
        "batch_size": args.batch_size,
        "root": args.root,
        "trained_model_name": args.trained_model_name,
    }
    wandb.init(
        project="nvda-ml-activity-detection",
        entity="ict3104-team14-2022",
        config=wandb.config,
        job_type="training",
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

    root = str(args.root)
    num_channel = int(args.num_channel)
    input_channel = 1024
    # Set classes, 51 for TSU.
    num_classes = 51

    # Printing
    print("batch_size:", batch_size)
    print("Cuda Available: ", torch.cuda.is_available())
    print("GPU with CUDA :", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    # Set Stream
    dataloaders, datasets = load_data(train_split, test_split, root, num_classes)

    # Add more models here if required.
    if args.model == "PDAN":
        print("you are processing PDAN")
        import models

        models = models.PDAN(
            num_stages=1,
            num_layers=5,
            num_f_maps=num_channel,
            dim=input_channel,
            num_classes=num_classes,
        )
        models = torch.nn.DataParallel(models)

        if args.load_model != "False":
            # entire model
            models = torch.load(args.load_model)
            # model.load_state_dict(torch.load(str(args.load_model)))
            print("loaded", args.load_model)

        pytorch_total_params = sum(
            p.numel() for p in models.parameters() if p.requires_grad
        )
        print("pytorch_total_params", pytorch_total_params)
        print(
            "num_channel:",
            num_channel,
            "input_channel:",
            input_channel,
            "num_classes:",
            num_classes,
        )
        models.cuda()

        criterion = nn.NLLLoss(reduction="none")
        lr = float(args.lr)
        print(lr)
        optimizer = optim.Adam(models.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=8, verbose=True
        )
        train_model(
            [(models, 0, dataloaders, optimizer, lr_sched, args.comp_info)],
            criterion,
            num_epochs=int(args.epoch),
            classes=num_classes,
        )
        print("Training done.")
    else:
        print("Please select a valid CNN model (PDAN)")
