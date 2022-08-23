from dataset_loader.load_matterport3d_dataset import Matterport3dDataset
from torch_geometric.loader import DataLoader
import inflect


def create_label_lists(dataset, verbose=False):
    engine = inflect.engine()

    labels, pl_labels = [], []

    for node_type_str, node_type in zip(["building", "room", "object"],
                                        dataset.labels_dict.keys()):
        if verbose:
            print("---------------------------------------------------------")
            print(node_type_str)
            print("---------------------------------------------------------")
        label_list = list(dataset.labels_dict[node_type])
        label_list = [label.replace("_", " ") for label in label_list]
        if verbose:
            print(label_list)

        pl_label_list = []
        for label in label_list:
            if "equipment" in label or "shelves" in label or "stairs" in label or "clothes" in label:
                pl_label_list.append(label)
            else:
                pl_label_list.append(engine.plural(label))

        if verbose:
            print(pl_label_list)
        labels.append(label_list)
        pl_labels.append(pl_label_list)

    return labels, pl_labels