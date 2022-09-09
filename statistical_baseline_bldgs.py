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


def stat_baseline_bldgs(use_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Matterport3dDataset('./mp_data/bldg_infer.pkl')

    labels, pl_labels = create_label_lists(dataset)
    building_list, room_list, object_list = labels
    building_list_pl, room_list_pl, object_list_pl = pl_labels

    building_list = ["house", "office complex", "spa resort"]
    building_list_pl = ["houses", "office complexes", "spa resorts"]

    if use_test:
        dataset = dataset.get_test_set()

    dataloader = DataLoader(dataset, batch_size=82)

    bldg_room_co = np.load("cooccurrency_matrices/bldg_room/building_room.npy")
    bldg_room_co += 1
    bldg_room_co /= np.sum(bldg_room_co, axis=0, keepdims=True)
    print(bldg_room_co)

    bldg_room_co = torch.tensor(bldg_room_co).to(device)

    batch = next(iter(dataloader))

    label = (
        batch.y[batch.building_mask],
        batch.y[batch.room_mask],
        batch.y[batch.object_mask],
    )

    y_room = F.one_hot(label[1]).type(torch.LongTensor)

    (
        room_building_edge_index,
        object_room_edge_index,
        room_edge_index,
        object_edge_index,
    ) = (
        batch.room_building_edge_index,
        batch.object_room_edge_index,
        batch.room_edge_index,
        batch.object_edge_index,
    )

    category_index_map = get_category_index_map(batch)

    excluded_idxs = torch.tensor([0, 1, 21, 26]).to(device)
    room_counts = torch.zeros([3, 27]).to(device)
    bldg_counts = torch.zeros(3).to(device)

    correct, total = 0, 0

    data_dict = {bldg_label: [0, 0] for bldg_label in building_list}

    for i in tqdm(range(len(label[0]))):
        mask = category_index_map[room_building_edge_index[1]] == i
        neighbor_dists = y_room[category_index_map[room_building_edge_index[0]
                                                   [mask]]].to(device)

        room_mask = torch.sum(neighbor_dists, 0)
        room_mask[excluded_idxs] = 0

        room_counts[label[0][i]] += room_mask
        bldg_counts[label[0][i]] += 1
        room_mask = torch.sum(neighbor_dists, 0) > 0

        room_mask[excluded_idxs] = 0

        room_dist = torch.prod(bldg_room_co[:, room_mask], dim=1)

        actual_label = building_list[label[0][i]]
        inferred_label = building_list[torch.argmax(room_dist).cpu()]

        total += 1
        data_dict[actual_label][1] += 1

        if actual_label == inferred_label:
            data_dict[actual_label][0] += 1
            correct += 1

    print("correct:", correct, "total:", total)
    print(data_dict)


if __name__ == "__main__":
    stat_baseline_bldgs(True)