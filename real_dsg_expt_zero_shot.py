from load_matterport3d_dataset import Matterport3dDataset
from datetime import datetime
from statistics import mean, stdev
from os import path, mkdir
import random
import pickle
import pandas as pd

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

room_list = [
    "kitchen",
    "bathroom",
    "hallway",
    "lounge",
    "stairwell",  #"office",
    "bedroom"
]

room_list_pl = [
    "kitchens",
    "bathrooms",
    "hallways",
    "lounges",
    "stairwells",  #"offices",
    "bedrooms",
]

object_list = [
    None, None, None, None, "bed", "cabinet", None, "chair", "shelf", "mirror",
    "table", "wardrobe", "bathtub", "railing", "drawers", "counter", "sink",
    "refrigerator", "stairs", "cushion", "pool table", "toilet", "stove",
    "computer", "television", "washing machine", "microwave", "dishwasher",
    None, None, None
]

object_list_pl = [
    None, None, None, None, "beds", "cabinets", None, "chairs", "shelves",
    "mirrors", "tables", "wardrobes", "bathtubs", "railing", "drawers",
    "counters", "sinks", "refrigerators", "stairs", "cushions", "pool tables",
    "toilets", "stoves", "computers", "televisions", "washing machines",
    "microwaves", "dishwashers", None, None, None
]

sidpac_dsg_map = {
    0: ['lounge;seminar room', {5, 7, 10, 15}],
    1: ['hallway', set()],
    2: ['hallway', set()],
    3: ['music room', {4, 7, 10}],
    4: ['hallway', {5, 10}],
    5: ['lounge;game room', {5, 7, 10, 24}],
    6: ['hallway', set()],
    7: ['hallway', {7}],
    8: ['stairwell', {13, 18}],
    9: ['stairwell', {13, 18}],
    10: ['stairwell', {13, 18}],
    11: ['stairwell', {18}],
    12: ['stairwell;hallway', set()],
    13: ['hallway', {7, 10, 17}],
    14: ['bedroom', {4, 5, 7, 8, 10, 11, 14}],
    15: ['kitchen', {5, 10, 22}],
    16: ['hallway', set()],
    17: ['lounge;hallway', {7, 10}],
    18: ['hallway', {5}],
    19: ['hallway', set()],
    20: ['hallway', set()],
    21: ['hallway', set()],
    22: ['stairwell', {13, 18}],
    23: ['stairwell', {18}],
    24: ['stairwell', {13, 18}],
    25: ['hallway', set()]
}

apartment_dsg_map = {
    0: ['dining room;kitchen', {5, 7, 8, 10, 18}],
    1: ['bedroom', {4, 5, 7, 8, 9, 11}],
    2: ['office', {5, 8, 10, 15}],
    3: ['bedroom', {4, 7, 8, 9}]
}

office_dsg_map = {
    0: ['hallway;office', {5, 7, 8, 10, 11, 14, 15}],
    1: ['office', {5, 7, 8, 10, 15, 23}],
    2: ['office', {7, 10, 11, 15}],
    3: ['conference room', {7, 10}],
    4: ['office', set()]
}

dsg_map_list = [sidpac_dsg_map, apartment_dsg_map, office_dsg_map]

scores = torch.tensor([
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.5078, -4.9141, 0.0000, -5.1680,
        -4.9961, -5.1484, -5.0039, -4.7148, -4.6406, -5.7656, -4.7031, -4.8789,
        -5.2227, -4.4961, -5.2695, -4.9609, -5.2539, -5.2969, -4.2109, -5.1250,
        -5.1367, -4.5039, -4.3906, -4.1211, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.1992, -5.1562, 0.0000, -5.4102,
        -5.2031, -4.4453, -5.7656, -4.2656, -3.8984, -5.5391, -4.7539, -5.1602,
        -4.7109, -5.0273, -5.3242, -4.7188, -5.3477, -4.3477, -5.0117, -5.5586,
        -5.2617, -4.5117, -5.0820, -4.5859, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.3555, -5.2305, 0.0000, -5.0859,
        -5.0195, -4.7812, -5.2930, -4.5195, -4.8516, -5.2031, -4.9023, -5.2695,
        -5.6211, -5.0625, -4.6836, -4.9883, -5.1680, -5.3320, -5.1211, -5.1055,
        -5.1406, -4.9414, -5.0938, -4.8711, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.2891, -5.5625, 0.0000, -4.5391,
        -5.4375, -4.8867, -4.9023, -4.5625, -4.6602, -5.6172, -5.1055, -5.2656,
        -5.7227, -5.0195, -5.4570, -4.5117, -4.5547, -5.2227, -4.9648, -5.0938,
        -4.4414, -4.8867, -5.1094, -4.9648, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.5117, -5.4883, 0.0000, -5.4141,
        -5.1875, -4.8828, -5.7031, -4.8125, -4.7500, -4.8242, -5.0117, -5.4336,
        -5.5312, -5.1172, -4.0508, -5.0625, -5.2500, -5.1562, -4.9805, -5.3203,
        -5.3008, -4.9375, -5.1562, -4.9648, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -5.5352, -4.9336, 0.0000, -4.6641,
        -4.8945, -4.9141, -4.8359, -4.3516, -4.8750, -5.6836, -4.4258, -4.8789,
        -5.7422, -4.8594, -5.2227, -4.8594, -5.1250, -5.2383, -4.9805, -4.2500,
        -4.9297, -4.9883, -4.8906, -4.8945, 0.0000, 0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, -4.3438, -4.8945, 0.0000, -4.9141,
        -4.8477, -4.5156, -5.2539, -3.9590, -4.3320, -5.4141, -4.3242, -5.1602,
        -5.6055, -4.8164, -4.8945, -4.4727, -5.2539, -5.0547, -4.8555, -4.8789,
        -4.7734, -4.6602, -4.8906, -4.8828, 0.0000, 0.0000, 0.0000
    ]
])
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
lm_model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,  # low_cpu_mem_usage=True
)
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


def eval_query(query):
    room_probs = torch.zeros(len(room_list))
    for i, room in enumerate(room_list):
        if room[0] in "aeiou":
            query_with_suffix = query + "n " + room + "."
        else:
            query_with_suffix = query + " " + room + "."
        room_probs[i] = negcrossentropy(query_with_suffix)
    return torch.exp(room_probs) / torch.sum(torch.exp(room_probs))


exp = torch.exp(scores)
probs = exp / torch.sum(exp, dim=0, keepdim=True)
inverse_perplexities = torch.exp(torch.sum(probs * torch.log(probs), dim=0))

k = 3

for dsg_map, name in zip(dsg_map_list,
                         ["sidpac_floor1_3", "uh2_apartment", "uh2_office"]):
    start_msg = "################## Starting: " + name + " ##################"
    print("#" * len(start_msg))
    print(start_msg)
    print("#" * len(start_msg))

    room_ids, labels, predicted_labels, queries = [], [], [], []

    for room_id in dsg_map.keys():
        objs_in_room = dsg_map[room_id][1]

        k_prime = min(k, len(objs_in_room))

        if k_prime == 0:
            #query_str = "A room containing nothing is called a"
            room_ids.append(room_id)
            labels.append(dsg_map[room_id][0])
            predicted_labels.append("-")
            queries.append("-")
            continue
        else:
            print("---------", room_id, "---------")
            one_hot_vec = torch.zeros(len(object_list))
            for idx in objs_in_room:
                one_hot_vec[idx] = 1

            room_ppls = one_hot_vec * inverse_perplexities

            highest_score_objs = torch.topk(room_ppls, k_prime).indices.numpy()

            query_str = "A room containing "
            if k_prime == 1:
                query_str += object_list_pl[
                    highest_score_objs[0]] + " is called a"
            elif k_prime == 2:
                query_str += object_list_pl[
                    highest_score_objs[0]] + " and " + object_list_pl[
                        highest_score_objs[1]] + " is called a"
            else:
                for i in range(k_prime - 1):
                    query_str += object_list_pl[highest_score_objs[i]] + ", "
                query_str += "and " + object_list_pl[
                    highest_score_objs[-1]] + " is called a"

        room_dist = eval_query(query_str)

        print(query_str)
        print("predicted:",
              room_list[torch.argmax(room_dist).to("cpu").numpy()], "-",
              "ground truth:", dsg_map[room_id][0])

        room_ids.append(room_id)
        labels.append(dsg_map[room_id][0])
        predicted_labels.append(
            room_list[torch.argmax(room_dist).to("cpu").numpy()])
        queries.append(query_str)

    results_dict = {
        "room_id": room_ids,
        "label": labels,
        "predicted_label": predicted_labels,
        "queries": queries
    }

    df = pd.DataFrame(data=results_dict)
    df.to_csv(os.path.join("./real_dsg_results", name + "_results.csv"))