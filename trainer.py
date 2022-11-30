import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import YoloV1
from loss import YoloLoss
from dataset import PascalDataset
from data_tranforms import create_transform
from utils import load_checkpoint, get_bboxes, mean_average_precision

IMG_PATH = 'archive/images'
LABEL_PATH = 'archive/labels'
EXAMPLE_CSV = 'archive/100examples.csv'
TRAIN_CSV = 'archive/train.csv'
TEST_CSV = 'archive/test.csv'
TRANSFORMATION = create_transform()

LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'

# try:
#     if torch.cuda.is_available():
#         DEVICE = 'cuda'
# except RuntimeError:
#     DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 16
WEIGHT_DECAY = 0
LEARNING_RATE = 2e-5
IOU_THRESHOLD = 0.5
THRESHOLD = 0.4


def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = loss_fn(output, y)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")



def main():
    model = YoloV1(split_size = 7, num_boxes = 2, num_classes = 20)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)



    train_dataset = PascalDataset(
        img_path = IMG_PATH,
        label_path = LABEL_PATH,
        csv = EXAMPLE_CSV,
        transformation = TRANSFORMATION
    )

    test_dataset = PascalDataset(
        img_path = IMG_PATH,
        label_path = LABEL_PATH,
        csv = TEST_CSV,
        transformation = TRANSFORMATION
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = 2,
        pin_memory = True,
        shuffle = True,
        drop_last = False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = 2,
        pin_memory = True,
        shuffle = True,
        drop_last = False
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            loader = train_loader,
            model = model,
            iou_threshold = IOU_THRESHOLD,
            threshold = THRESHOLD,
            device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes,
            target_boxes, 
            iou_threshold=IOU_THRESHOLD, 
            box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")

        train(train_loader, model, optimizer, loss_fn)







if __name__ == "__main__":
    main()