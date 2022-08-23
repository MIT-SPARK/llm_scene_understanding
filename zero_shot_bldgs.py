from cProfile import label
from load_matterport3d_dataset import Matterport3dDataset
from statistics import mean, stdev
from os import path, mkdir
from extract_labels import create_label_lists
import torch
import torch_geometric
import gensim
from torch_geometric.loader import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from model_utils import get_category_index_map
import torch.nn.functional as F
from torch_scatter import scatter
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    RobertaTokenizer,
    RobertaForMaskedLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTJForCausalLM,
)
from tqdm import tqdm
from perplexity_measure import compute_object_norm_inv_ppl
import pandas as pd
import os


def zero_shot_bldgs(lm,
                    use_cooccurencies=True,
                    batch_size=82,
                    k=5,
                    label_set="nyuClass",
                    use_test=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if lm == "RoBERTa-large":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        lm_model = RobertaForMaskedLM.from_pretrained("roberta-large")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set +
            "_roberta_large/building_room.npy")
        start = "<s>"
        end = "</s>"
        mask_id = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<mask>"))[0]
    elif lm == "GPT-J":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        lm_model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,  # low_cpu_mem_usage=True
        )
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set +
            "_gpt_j/building_room.npy")
    else:
        print("Model option " + lm + " not implemented yet")
        raise NotImplemented

    if use_cooccurencies:
        """ object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set + "_gt/building_room.npy",
            True) """
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/norm_bldg_room/building_room.npy", True)

    object_norm_inv_perplexity = object_norm_inv_perplexity.to(device)
    lm_model.eval()
    lm_model.to(device)

    def negcrossentropy(text):
        tokens_tensor = tokenizer.encode(text,
                                         add_special_tokens=False,
                                         return_tensors="pt").to(device)
        with torch.no_grad():
            output = lm_model(tokens_tensor, labels=tokens_tensor)
            loss = output[0]

            return -loss

    def pll(text, mean=True):
        # Tokenize input
        text = start + " " + text.capitalize() + " " + end
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).reshape(-1)
        masked_tokens_tensor = tokens_tensor.repeat(tokens_tensor.shape[0] - 3,
                                                    1)

        # Mask out one token per row
        ind1 = np.arange(masked_tokens_tensor.shape[0])
        ind2 = np.arange(1, masked_tokens_tensor.shape[0] + 1)
        masked_tokens_tensor[ind1, ind2] = mask_id

        masked_tokens_tensor = masked_tokens_tensor.to(
            device)  # if you have gpu

        # Predict all tokens
        with torch.no_grad():
            total = 0
            for i in range(len(masked_tokens_tensor)):
                outputs = lm_model(masked_tokens_tensor[i].unsqueeze(0))
                predictions = outputs[0]  # .to("cpu")
                mask_scores = predictions[0, i + 1]
                total += mask_scores[tokens_tensor[i + 1]] - torch.logsumexp(
                    mask_scores, dim=0)

        Z = len(masked_tokens_tensor) if mean else 1

        return total / Z

    if lm in ["GPT2-large", "GPT-Neo", "GPT-J"]:
        scoring_fxn = negcrossentropy
    else:
        scoring_fxn = pll

    def construct_dist(rooms, scoring_fxn, print_query=False):
        query_str = "A building containing "
        names = []
        for rm in rooms:
            names.append(room_list_pl[rm])
        if len(names) == 1:
            query_str += names[0]
        elif len(names) == 2:
            query_str += names[0] + " and " + names[1]
        else:
            for name in names[:-1]:
                query_str += name + ", "
            query_str += "and " + names[-1]
        query_str += " is called a"

        if print_query:
            print(query_str)

        TEMP = []
        for bldg in building_list:
            TEMP_STR = query_str + "n " if bldg[
                0] in "aeiou" else query_str + " "
            TEMP_STR += bldg + "."
            score = scoring_fxn(TEMP_STR)
            TEMP.append(score)
        dist = torch.tensor(TEMP)

        if lm == "GPT-J":
            dist = dist.type(torch.DoubleTensor)
        return dist

    dataset = Matterport3dDataset('./mp_data/bldg_infer.pkl')

    labels, pl_labels = create_label_lists(dataset)
    building_list, room_list, object_list = labels
    building_list_pl, room_list_pl, object_list_pl = pl_labels

    building_list = ["house", "office complex", "spa resort"]
    building_list_pl = ["houses", "office complexes", "spa resorts"]

    if use_test:
        dataset = dataset.get_test_set()

    # create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size)

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

        present_rooms_scores = room_mask * object_norm_inv_perplexity

        top_k_rooms = torch.topk(present_rooms_scores, k).indices

        best_bldg_idx = torch.argmax(
            construct_dist(top_k_rooms,
                           scoring_fxn,
                           print_query=label[0][i] == 2)).to("cpu")
        if label[0][i] == 2:
            print(
                "--------------------------------------------------------------"
            )
            print(building_list[label[0][i]])
            print(building_list[best_bldg_idx])

        if label[0][i] == best_bldg_idx:
            correct += 1
            data_dict[building_list[label[0][i]]][0] += 1

        total += 1
        data_dict[building_list[label[0][i]]][1] += 1

    for i in torch.sort(object_norm_inv_perplexity, descending=True).indices:
        print(room_list[i])
    print(room_list)

    print("Correct:", correct)
    print("Total:", total)
    print(data_dict)


if __name__ == "__main__":
    # Can change label set to nyuClass
    zero_shot_bldgs("GPT-J", True, label_set="mpcat40", k=4, use_test=False)