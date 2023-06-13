import os
import pickle
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from dataset import FinetuningDataset
from models import ContrastiveNet, FeedforwardNet


def train_job(lm,
              label_set,
              epochs,
              batch_size,
              holdout_objs,
              holdout_rooms,
              co_suffix="",
              seed=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def ff_loss(pred, label):
        return F.cross_entropy(pred, label)

    # Create datasets
    ds = FinetuningDataset(lm, label_set, co_suffix)
    train_ds, val_ds, test_ds, holdout_ds = ds.create_holdout_split(
        0.4, 0.2, holdout_objs, holdout_rooms)

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    holdout_dl = DataLoader(holdout_ds, batch_size=1)

    output_size = len(train_ds.room_list)

    ff_net = FeedforwardNet(1024, output_size)
    ff_net.to(device)

    optimizer = torch.optim.Adam(ff_net.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.001)

    loss_fxn = ff_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.99)

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    desc = ""
    torch.manual_seed(seed)
    with trange(epochs) as pbar:
        for epoch in pbar:
            train_epoch_loss = []
            val_epoch_loss = []
            train_epoch_acc = []
            val_epoch_acc = []
            for batch_idx, (query_em, _, label) in enumerate(train_dl):
                pred = ff_net(query_em)

                loss = loss_fxn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

                accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
                train_epoch_acc.append(accuracy)

                if batch_idx % 100 == 0:
                    pbar.set_description((desc).rjust(20))

            scheduler.step()
            train_losses.append(torch.mean(torch.tensor(train_epoch_loss)))
            train_acc.append(torch.mean(torch.tensor(train_epoch_acc)))

            for batch_idx, (query_em, _, label) in enumerate(val_dl):
                with torch.no_grad():
                    pred = ff_net(query_em)
                    loss = loss_fxn(pred, label)
                    val_epoch_loss.append(loss.item())

                    accuracy = ((torch.argmax(pred, dim=1) == label) *
                                1.0).mean()
                    val_epoch_acc.append(accuracy)
                    if batch_idx % 100 == 0:
                        desc = (f"{np.mean(np.array(train_epoch_loss)):6.4}" +
                                ", " + f"{accuracy.item():6.4}")
                        pbar.set_description((desc).rjust(20))
            val_losses.append(torch.mean(torch.tensor(val_epoch_loss)))
            val_acc.append(torch.mean(torch.tensor(val_epoch_acc)))

    ff_net.eval()

    test_loss, test_acc = [], []
    for batch_idx, (query_em, _, label) in enumerate(test_dl):
        with torch.no_grad():
            pred = ff_net(query_em)
            loss = loss_fxn(pred, label)
            test_loss.append(loss.item())

            accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
            test_acc.append(accuracy)

    holdout_loss, holdout_acc_dict = [], {
        term: []
        for term in holdout_ds.holdout_terms
    }
    holdout_total_acc = []
    for batch_idx, (query_em, _, label, term) in enumerate(holdout_dl):
        with torch.no_grad():
            pred = ff_net(query_em)
            loss = loss_fxn(pred, label)
            holdout_loss.append(loss.item())

            accuracy = (torch.argmax(pred, dim=1) == label) * 1.0
            holdout_acc_dict[term[0]].append(accuracy)
            holdout_total_acc.append(accuracy)

    return (
        train_losses,
        val_losses,
        train_acc,
        val_acc,
        test_loss,
        test_acc,
        holdout_acc_dict,
    )


if __name__ == "__main__":
    lm = "RoBERTa-large"
    label_set = "nyuClass"
    use_gt = True
    holdout_objs, holdout_rooms = [
        "beds",
        "toilets",
        "sink",
        "washing machines",
        "chair",
    ], []

    print("Starting:", lm, label_set, "use_gt =", use_gt)
    co_suffix = "_useGT_True_502030" if use_gt else "_gpt_j_co"

    torch.manual_seed(0)

    (
        train_losses,
        val_losses,
        train_acc,
        val_acc,
        test_loss,
        test_acc,
        holdout_acc_dict,
    ) = train_job(lm,
                  label_set,
                  100,
                  512,
                  holdout_objs,
                  holdout_rooms,
                  co_suffix=co_suffix)
    for key in holdout_acc_dict.keys():
        print(key)
        print(torch.mean(torch.tensor(holdout_acc_dict[key])))
    exit()
    pickle.dump(holdout_acc_dict,
                open("./ff_holdout_results/holdout_acc_dict.pkl", "wb"))
