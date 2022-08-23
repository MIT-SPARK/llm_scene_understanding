from dataset_loader.load_matterport3d_dataset import Matterport3dDataset
from extract_labels import *
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import os
from tqdm import tqdm

#################################################################################
dataset_name = "mpcat40"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 82

dataset = Matterport3dDataset('./mp_data/matterport3d/' + dataset_name +
                              '_matterport3d_w_edge.pkl')

labels, pl_labels = create_label_lists(dataset)
building_list, room_list, object_list = labels
building_list_pl, room_list_pl, object_list_pl = pl_labels

# create data loader
dataloader = DataLoader(dataset, batch_size=batch_size)

batch = next(iter(dataloader))

building_room_co_mtx = np.zeros([len(building_list), len(room_list)])
room_room_co_mtx = np.zeros([len(room_list), len(room_list)])
room_object_co_mtx = np.zeros([len(room_list), len(object_list)])
object_object_co_mtx = np.zeros([len(object_list), len(object_list)])

# Create predictions and labels for batch
with torch.no_grad():
    y = batch.y

    # Each of these tensors is size [2, # edges of given type]. Name describes two nodes in each edge,
    # e.g. room_building means one is a room and other is building
    room_building_edge_index, object_room_edge_index, room_edge_index, object_edge_index = \
                batch.room_building_edge_index, batch.object_room_edge_index, batch.room_edge_index, batch.object_edge_index

    # Room / Bldg
    rb_room_labels = y[room_building_edge_index[0]]
    rb_bldg_labels = y[room_building_edge_index[1]]
    print("Starting building / room matrix...")
    for i, j in tqdm(zip(rb_bldg_labels, rb_room_labels)):
        building_room_co_mtx[i, j] += 1

    # Room / Object
    or_obj_labels = y[object_room_edge_index[0]]
    or_room_labels = y[object_room_edge_index[1]]
    print("Starting room / object matrix...")
    for i, j in tqdm(zip(or_room_labels, or_obj_labels)):
        room_object_co_mtx[i, j] += 1

    # Room / Room
    rr_labels_1 = y[room_edge_index[0]]
    rr_labels_2 = y[room_edge_index[1]]
    print("Starting room / room matrix...")
    for i, j in tqdm(zip(rr_labels_1, rr_labels_2)):
        room_room_co_mtx[i, j] += 1

    rr_co_mtx_TEMP = np.copy(room_room_co_mtx).T
    rr_co_mtx_TEMP[:, np.arange(len(room_list))] = 0
    room_room_co_mtx += rr_co_mtx_TEMP

    # Room / Room
    oo_labels_1 = y[object_edge_index[0]]
    oo_labels_2 = y[object_edge_index[1]]
    print("Starting object / object matrix...")
    for i, j in tqdm(zip(oo_labels_1, oo_labels_2)):
        object_object_co_mtx[i, j] += 1

    oo_co_mtx_TEMP = np.copy(object_object_co_mtx).T
    oo_co_mtx_TEMP[:, np.arange(len(object_list))] = 0
    object_object_co_mtx += oo_co_mtx_TEMP

name_mtx_list = [("building_room", building_room_co_mtx),
                 ("room_object", room_object_co_mtx),
                 ("room_room", room_room_co_mtx),
                 ("object_object", object_object_co_mtx)]
for name, arr in name_mtx_list:
    path = os.path.join("./cooccurrency_matrices", dataset_name + "_gt",
                        name + ".npy")
    np.save(path, arr)