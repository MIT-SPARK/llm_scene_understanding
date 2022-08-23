import numpy as np
from labels import *
import torch
from torch.nn import functional as F


def compute_object_norm_inv_ppl(path_to_room_object_mtx,
                                use_cooccurrencies=False):
    room_object = torch.tensor(np.load(path_to_room_object_mtx))
    object_room = room_object.T
    if use_cooccurrencies:
        object_room += 1. / object_room.shape[1]  # Smoothing
        object_room_norm = object_room / torch.sum(
            object_room, dim=1, keepdim=True)
    else:
        object_room_norm = F.softmax(object_room, dim=1)
    object_entropy = -torch.sum(
        object_room_norm * torch.log(object_room_norm), dim=1, keepdim=False)

    object_norm_inv_perplexity = F.softmax(-object_entropy, dim=0)
    return object_norm_inv_perplexity