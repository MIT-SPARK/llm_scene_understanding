import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


def get_category_index_map(torch_data):
    """
    Compute a 1-dim tensor mapping node index in torch_data to type specific node index.
    """
    room_building_edge_index, object_room_edge_index, room_edge_index, object_edge_index, \
    building_mask, room_mask, object_mask = \
        torch_data.room_building_edge_index, torch_data.object_room_edge_index, torch_data.room_edge_index, \
        torch_data.object_edge_index, torch_data.building_mask, torch_data.room_mask, torch_data.object_mask

    category_index_map = torch.zeros(torch_data.num_nodes, dtype=torch.int64)
    category_index_map[building_mask] = torch.arange(
        building_mask.sum().item())
    category_index_map[room_mask] = torch.arange(room_mask.sum().item())
    category_index_map[object_mask] = torch.arange(object_mask.sum().item())
    return category_index_map.to(building_mask.device)
