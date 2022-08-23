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

from dataset import create_building_splits, create_comparison_building_splits
from models import ContrastiveNet, FeedforwardNet


def train_job(lm, epochs, batch_size, co_suffix="", seed=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def ff_loss(pred, label):
        return F.cross_entropy(pred, label)

    # Create datasets
    train_ds, val_ds, test_ds = create_comparison_building_splits(
        "./building_data/comparison_data/" + lm + "_gt", device=device)

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    output_size = len(train_ds.building_list)

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
            for batch_idx, (query_em, label) in enumerate(train_dl):
                pred = ff_net(query_em)

                loss = loss_fxn(pred, label)

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

            for batch_idx, (query_em, label) in enumerate(val_dl):
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

    test_loss, test_acc = [], []

    test_acc_by_class = {bldg: [0, 0] for bldg in test_ds.building_list}
    for batch_idx, (query_em, label) in enumerate(test_dl):
        with torch.no_grad():
            pred = ff_net(query_em)
            loss = loss_fxn(pred, label)
            test_loss.append(loss.item())

            accuracy = ((torch.argmax(pred, dim=1) == label) * 1.0).mean()
            test_acc.append(accuracy)

            for bldg_idx, bldg in enumerate(test_ds.building_list):
                bldg_mask = label == bldg_idx
                num_bldg = (bldg_mask * 1).sum()
                num_corr = (
                    (torch.argmax(pred, dim=1)[bldg_mask] == bldg_idx) *
                    1).sum()

                test_acc_by_class[bldg][0] += num_corr
                test_acc_by_class[bldg][1] += num_bldg

    print("test loss:", torch.mean(torch.tensor(test_loss)))
    print("test acc:", torch.mean(torch.tensor(test_acc)))
    print(test_acc_by_class)

    return train_losses, val_losses, train_acc, val_acc, test_loss, test_acc


if __name__ == "__main__":
    (
        train_losses_list,
        val_losses_list,
        train_acc_list,
        val_acc_list,
        test_loss_list,
        test_acc_list,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for lm in ["RoBERTa-large", "BERT-large"]:

        print("Starting:", lm)
        co_suffix = "_gt"
        (
            train_losses,
            val_losses,
            train_acc,
            val_acc,
            test_loss,
            test_acc,
        ) = train_job(lm, 100, 512, co_suffix=co_suffix)
        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    pickle.dump(
        train_losses_list,
        open("./bldg_ff_results/comparison_results/train_losses.pkl", "wb"))
    pickle.dump(
        train_acc_list,
        open("./bldg_ff_results/comparison_results/train_acc.pkl", "wb"))
    pickle.dump(
        val_losses_list,
        open("./bldg_ff_results/comparison_results/val_losses.pkl", "wb"))
    pickle.dump(val_acc_list,
                open("./bldg_ff_results/comparison_results/val_acc.pkl", "wb"))
    pickle.dump(
        test_loss_list,
        open("./bldg_ff_results/comparison_results/test_loss.pkl", "wb"))
    pickle.dump(
        test_acc_list,
        open("./bldg_ff_results/comparison_results/test_acc.pkl", "wb"))
