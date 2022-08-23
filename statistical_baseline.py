from cProfile import label
import os
from tqdm import tqdm
from load_matterport3d_dataset import Matterport3dDataset
from model_utils import get_category_index_map
from perplexity_measure import compute_object_norm_inv_ppl
from extract_labels import create_label_lists
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pickle

import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    GPTNeoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTJModel,
)


class BaselineRunner:

    def __init__(self, device=None, verbose=False, label_set="mpcat40"):

        self.verbose = verbose
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None else device)

        dataset = Matterport3dDataset("./mp_data/" + label_set +
                                      "_matterport3d_w_edge.pkl")
        labels, pl_labels = create_label_lists(dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        if self.verbose:
            print("Using device:", self.device)

        # create data loader
        self.dataloader = DataLoader(dataset, batch_size=82)
        room_obj_freqs = (np.load(
            "./cooccurrency_matrices/" + label_set + "_gt" +
            "/room_object.npy", ) + 1)
        self.object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set + "_gt" +
            "/room_object.npy",
            True,
        ).to(self.device)

        excluded_room_indices = np.array([
            self.room_list.index(excluded_room)
            for excluded_room in ["None", "yard", "porch", "balcony"]
        ])
        room_obj_freqs[excluded_room_indices] = 0

        self.room_obj_conditionals = room_obj_freqs / np.sum(
            room_obj_freqs, axis=0, keepdims=True)
        """ print(self.room_obj_conditionals) """

    def extract_data(self, max_num_obj):
        """
        Extracts and saves the most interesting objects from each room.

        TODO: Finish docstring
        """
        # self.max_num_obj = max_num_obj

        batch = next(iter(self.dataloader))
        label = (
            batch.y[batch.building_mask],
            batch.y[batch.room_mask],
            batch.y[batch.object_mask],
        )
        y_object = F.one_hot(label[-1]).type(torch.LongTensor)
        category_index_map = get_category_index_map(batch)
        object_room_edge_index = batch.object_room_edge_index

        total_count = 0
        correct_count = 0
        correct_count_use_topk = 0

        for i in range(len(label[1])):  # range(len(label[1])):
            ground_truth_room = label[1][i]

            mask = category_index_map[object_room_edge_index[1]] == i
            neighbor_dists = y_object[category_index_map[
                object_room_edge_index[0][mask]]]
            neighbor_dists = neighbor_dists.to(self.device)
            all_objs = torch.sum(neighbor_dists, dim=0) > 0

            scores = all_objs * self.object_norm_inv_perplexity
            best_objs = (torch.topk(
                scores, max(min((all_objs > 0).sum(), max_num_obj),
                            1)).indices.cpu().numpy().flatten())

            room_label = self.room_list[ground_truth_room]
            if (room_label in ["None", "yard", "porch", "balcony"]
                    or len(neighbor_dists) == 0):
                continue

            # print("------------------------------------------------")
            # print(self.room_list[ground_truth_room])
            objs_list = all_objs.nonzero().cpu().numpy().flatten()
            inferred_room_dist = self.room_obj_conditionals[:, objs_list].prod(
                axis=1)
            inferred_room = np.argmax(inferred_room_dist)

            inferred_room_dist_use_topk = self.room_obj_conditionals[:,
                                                                     best_objs].prod(
                                                                         axis=1
                                                                     )
            inferred_room_use_topk = np.argmax(inferred_room_dist_use_topk)
            # print(self.room_list[inferred_room])

            if inferred_room == ground_truth_room:
                correct_count += 1
            if inferred_room_use_topk == ground_truth_room:
                correct_count_use_topk += 1
            total_count += 1

        print("Fraction correct using all objects:",
              correct_count / total_count)
        print(
            "Fraction correct using top k objects:",
            correct_count_use_topk / total_count,
        )


if __name__ == "__main__":
    for label_set in ["mpcat40", "nyuClass"]:
        print(label_set)
        bl_runner = BaselineRunner(label_set=label_set)
        bl_runner.extract_data(3)
