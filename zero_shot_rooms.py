from cProfile import label
from load_matterport3d_dataset import Matterport3dDataset
from datetime import datetime
from statistics import mean, stdev
from os import path, mkdir
import random
import pickle

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


#################################################################################
def dynamic_lm_refinements(
        lm,
        use_cooccurencies=True,
        batch_size=82,
        k=5,
        use_test=False,
        label_set="nyuClass",  # "mpcat40",
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if lm == "BERT":
        raise NotImplemented
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        lm_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./label_matrices/bert_base_mean_pll/room_object.npy")
        start = "[CLS]"
        end = "[SEP]"
        mask_id = 103
    elif lm == "BERT-large":
        raise NotImplemented
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        lm_model = BertForMaskedLM.from_pretrained("bert-large-uncased")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./label_matrices/bert_large_mean_pll/room_object.npy")
        start = "[CLS]"
        end = "[SEP]"
        mask_id = 103
    elif lm == "RoBERTa":
        raise NotImplemented
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        lm_model = RobertaForMaskedLM.from_pretrained("roberta-base")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./label_matrices/roberta_base_mean_pll/room_object.npy")
        start = "<s>"
        end = "</s>"
        mask_id = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<mask>"))[0]
    elif lm == "RoBERTa-large":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        lm_model = RobertaForMaskedLM.from_pretrained("roberta-large")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set +
            "_roberta_large/room_object.npy")
        start = "<s>"
        end = "</s>"
        mask_id = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<mask>"))[0]
    elif lm == "GPT2-large":
        raise NotImplemented
        lm_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./label_matrices/gpt2_large_negcrossent/room_object.npy")
    elif lm == "GPT-Neo":
        raise NotImplemented
        lm_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./label_matrices/gptneo_negcrossent/room_object.npy")
    elif lm == "GPT-J":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        lm_model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,  # low_cpu_mem_usage=True
        )
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set + "_gpt_j/room_object.npy")
    else:
        print("Model option " + lm + " not implemented yet")
        raise NotImplemented

    if use_cooccurencies:
        object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            "./cooccurrency_matrices/" + label_set + "_gt/room_object.npy",
            True)

    object_norm_inv_perplexity = object_norm_inv_perplexity.to(device)

    lm_model.eval()
    lm_model.to(device)

    #################################################################################
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

    dataset = Matterport3dDataset(
        "./mp_data/" + label_set + "_matterport3d_w_edge.pkl"
    )  # TODO: Change back out of _502030 if needed

    labels, pl_labels = create_label_lists(dataset)
    building_list, room_list, object_list = labels
    building_list_pl, room_list_pl, object_list_pl = pl_labels

    if use_test:
        dataset = dataset.get_test_set()

    # create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    def construct_dist(objs, scoring_fxn):
        query_str = "A room containing "
        names = []
        for ob in objs:
            names.append(object_list_pl[ob])
        if len(names) == 1:
            query_str += names[0]
        elif len(names) == 2:
            query_str += names[0] + " and " + names[1]
        else:
            for name in names[:-1]:
                query_str += name + ", "
            query_str += "and " + names[-1]
        query_str += " is called a"

        TEMP = []
        for room in room_list:
            if room in ["None", "balcony", "porch", "yard"]:
                TEMP.append(-float("inf"))
            else:
                if room[0] in "aeiou" and room != "utility room":
                    TEMP_STR = query_str + "n "
                else:
                    TEMP_STR = query_str + " "
                TEMP_STR += room + "."
                score = scoring_fxn(TEMP_STR)
                TEMP.append(score)
        dist = torch.tensor(TEMP)

        if lm == "GPT-J":
            dist = dist.type(torch.DoubleTensor)
        return dist

    batch = next(iter(dataloader))

    df = pd.DataFrame(columns=["objects", "predicted room", "actual room"])

    # Create predictions and labels for batch
    with torch.no_grad():
        label = (
            batch.y[batch.building_mask],
            batch.y[batch.room_mask],
            batch.y[batch.object_mask],
        )
        y_object = F.one_hot(label[-1],
                             len(object_list)).type(torch.LongTensor)

        # Each of these tensors is size [2, # edges of given type]. Name describes two nodes in each edge,
        # e.g. room_building means one is a room and other is building
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

        # Integer tensor of size [# nodes], where each node is given a unique number among the nodes of same type,
        # e.g. if nodes are of nodetype [B, B, R, O, R, O, B], then returned tensor is [0, 1, 0, 0, 1, 1, 2]
        category_index_map = get_category_index_map(batch)

        ref_correct = 0
        total = 0

        for i in tqdm(range(len(label[1]))):  # range(len(label[1])):
            # print(room_list[label[1][i]])
            mask = category_index_map[object_room_edge_index[1]] == i
            neighbor_dists = y_object[category_index_map[
                object_room_edge_index[0][mask]]]
            neighbor_dists = neighbor_dists.to(device)
            if len(neighbor_dists) == 0:
                continue
            scores = neighbor_dists * object_norm_inv_perplexity.reshape(
                [1, -1])

            maxes = torch.max(scores, dim=1).values
            top_max_inds = torch.topk(maxes, max(min((maxes > 0).sum(), k),
                                                 1)).indices
            objs = torch.argmax(scores[top_max_inds], dim=1)
            objs = torch.where(
                torch.bincount(objs, minlength=len(object_list)) > 0)[0]

            ref_dist = F.softmax(construct_dist(objs, scoring_fxn),
                                 dim=0).to(device)
            new_dist = ref_dist

            ref = torch.argmax(new_dist)
            actual = label[1][i]

            if ref == actual:
                ref_correct += 1
            total += 1
            object_strings = [object_list[id] for id in objs]

            df2 = pd.DataFrame(
                [[object_strings, room_list[ref], room_list[actual]]],
                columns=["objects", "predicted room", "actual room"],
            )
            df = pd.concat([df, df2], ignore_index=True)
        print("Accuracy:", ref_correct / total)

        return (ref_correct / total, df)


if __name__ == "__main__":

    lms = ["GPT-J"]

    df = pd.DataFrame(columns=["lm", "cooccurrencies", "accuracy"])

    for use_test in [True, False]:
        for label_set in ["nyuClass", "mpcat40"]:
            for co in [True, False]:
                for lm in lms:
                    print("label set:", label_set, "lm:", lm,
                          "use cooccurrencies:", co, "use test:", use_test)
                    acc, analysis_df = dynamic_lm_refinements(
                        lm,
                        use_cooccurencies=co,
                        batch_size=82,
                        k=3,
                        use_test=use_test,
                        label_set=label_set,
                    )
                    df2 = {
                        "lm": lm,
                        "cooccurrencies": co,
                        "accuracy": acc,
                    }
                    print(df2)
                    df = df.append(df2, ignore_index=True)

                    df.to_csv("./results/" + label_set + "_results_new.csv")
                    analysis_df.to_csv(
                        os.path.join(
                            "./analysis_logs",
                            lm + "_co_" + str(co) + "_labelset_" + label_set +
                            "_usetestset_" + str(use_test) + ".csv",
                        ))
