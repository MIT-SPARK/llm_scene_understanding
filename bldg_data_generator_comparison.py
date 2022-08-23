from cProfile import label
import os
from random import sample
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


class DataGenerator:

    def __init__(
        self,
        default_lm=None,
        device=None,
        verbose=False,
        label_set="mpcat40",
        use_gt_cooccurrencies=True,
    ):

        self.verbose = verbose
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None else device)
        self.dataset = Matterport3dDataset('./mp_data/bldg_infer.pkl')
        labels, pl_labels = create_label_lists(self.dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        self.excluded_rooms = ["None", "yard", "porch", "balcony"]

        if self.verbose:
            print("Using device:", self.device)

        path_to_cooccurrencies = (
            "./cooccurrency_matrices/norm_bldg_room/building_room.npy")

        self.object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            path_to_cooccurrencies,
            use_gt_cooccurrencies,
        ).to(self.device)

        self.cooccurrencies = np.load(path_to_cooccurrencies)
        self.cooccurrencies /= np.sum(self.cooccurrencies,
                                      axis=1,
                                      keepdims=True)

        self.lm = None
        self.lm_model = None
        self.tokenizer = None
        self.embedder = None

        if default_lm is not None:
            self.configure_lm(default_lm)

        self.rooms = {"train": [], "val": [], "test": []}
        self.labels = {"train": [], "val": [], "test": []}

    def configure_lm(self, lm):
        """
        Configure the language model, tokenizer, and embedding generator function.

        Sets self.lm, self.lm_model, self.tokenizer, and self.embedder based on the
        selected language model inputted to this function.

        Args:
            lm: str representing name of LM to use

        Returns:
            None
        """
        if self.lm is not None and self.lm == lm:
            print("LM already set to", lm)
            return

        self.lm = lm

        if self.verbose:
            print("Setting up LM:", self.lm)

        if lm == "BERT":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            lm_model = BertModel.from_pretrained("bert-base-uncased")
            start = "[CLS]"
            end = "[SEP]"
        elif lm == "BERT-large":
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            lm_model = BertModel.from_pretrained("bert-large-uncased")
            start = "[CLS]"
            end = "[SEP]"
        elif lm == "RoBERTa":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            lm_model = RobertaModel.from_pretrained("roberta-base")
            start = "<s>"
            end = "</s>"
        elif lm == "RoBERTa-large":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            lm_model = RobertaModel.from_pretrained("roberta-large")
            start = "<s>"
            end = "</s>"
        elif lm == "GPT2-large":
            lm_model = GPT2Model.from_pretrained("gpt2-large")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        elif lm == "GPT-Neo":
            lm_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
            tokenizer = GPT2Tokenizer.from_pretrained(
                "EleutherAI/gpt-neo-1.3B")
        elif lm == "GPT-J":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            lm_model = GPTJModel.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                torch_dtype=torch.float16,  # low_cpu_mem_usage=True
            )
        else:
            print("Model option " + lm + " not implemented yet")
            raise

        self.lm_model = lm_model
        self.lm_model.eval()
        self.lm_model = self.lm_model.to(self.device)

        self.tokenizer = tokenizer

        if self.verbose:
            print("Loaded LM:", self.lm)
        # self.tokenizer = self.tokenizer.to(self.device)

        if lm in ["BERT", "BERT-large", "RoBERTa", "RoBERTa-large"]:
            self.embedder = self._initialize_embedder(True,
                                                      start=start,
                                                      end=end)
        else:
            self.embedder = self._initialize_embedder(False)

        if self.verbose:
            print("Created corresponding embedder.")

        return

    def _initialize_embedder(self, is_mlm, start=None, end=None):
        """
        Returns a function that embeds sentences with the selected
        language model.

        Args:
            is_mlm: bool (optional) indicating if self.lm_model is an mlm.
                Default
            start: str representing start token for MLMs.
                Must be set if is_mlm == True.
            end: str representing end token for MLMs.
                Must be set if is_mlm == True.

        Returns:
            function that takes in a query string and outputs a
                [batch size=1, hidden state size] summary embedding
                using self.lm_model
        """
        if not is_mlm:

            def embedder(query_str):
                tokens_tensor = torch.tensor(
                    self.tokenizer.encode(query_str,
                                          add_special_tokens=False,
                                          return_tensors="pt").to(self.device))

                outputs = self.lm_model(tokens_tensor)
                print(outputs)
                print(outputs.last_hidden_state.shape)
                # Shape (batch size=1, hidden state size)
                return outputs.last_hidden_state[:, -1]

        else:

            def embedder(query_str):
                query_str = start + " " + query_str + " " + end
                tokenized_text = self.tokenizer.tokenize(query_str)
                tokens_tensor = torch.tensor(
                    [self.tokenizer.convert_tokens_to_ids(tokenized_text)])
                """ tokens_tensor = torch.tensor([indexed_tokens.to(self.device)])
                 """
                tokens_tensor = tokens_tensor.to(
                    self.device)  # if you have gpu

                with torch.no_grad():
                    outputs = self.lm_model(tokens_tensor)
                    # hidden state is a tuple
                    hidden_state = outputs.last_hidden_state

                # Shape (batch size=1, num_tokens, hidden state size)
                # Return just the start token's embeddinge
                return hidden_state[:, -1]

        return embedder

    def extract_data(self, num_samples, num_rooms_per_bldg):
        """
        Extracts and saves the most interesting objects from each room.

        TODO: Finish docstring
        """

        for split, split_fxn in (["train", self.dataset.get_training_set
                                  ], ["val", self.dataset.get_validation_set],
                                 ["test", self.dataset.get_test_set]):

            print(
                "#############################################################"
            )
            print(split)

            dataloader = DataLoader(split_fxn(), batch_size=82)
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

            excluded_idxs = torch.tensor([0, 1, 21, 26]).to(self.device)

            object_norm_inv_perplexity = self.object_norm_inv_perplexity.cpu(
            ).numpy()

            for i in tqdm(range(len(label[0]))):
                bldg_label = label[0][i]

                mask = category_index_map[room_building_edge_index[1]] == i
                neighbor_dists = y_room[category_index_map[
                    room_building_edge_index[0][mask]]].to(self.device)

                room_count = torch.sum(neighbor_dists, 0)
                room_count[excluded_idxs] = 0
                room_dist = room_count.cpu().float().numpy()

                room_dist /= np.sum(room_dist)

                if split != "test":
                    for i in range(num_samples):
                        chosen_rooms = np.random.choice(
                            len(room_dist),
                            size=num_rooms_per_bldg,
                            replace=False,
                            p=room_dist)
                        np.random.shuffle(chosen_rooms)

                        self.rooms[split].append(chosen_rooms)
                        self.labels[split].append(bldg_label)
                else:
                    room_mask = room_dist > 0
                    scores = room_mask * object_norm_inv_perplexity
                    chosen_rooms = np.argsort(
                        scores)[::-1][:num_rooms_per_bldg]

                    self.rooms[split].append(chosen_rooms)
                    self.labels[split].append(bldg_label)

    def generate_data(self):
        query_sentence_dict = {"train": [], "val": [], "test": []}
        label_dict = {"train": [], "val": [], "test": []}
        query_embedding_dict = {"train": [], "val": [], "test": []}

        for split in ["train", "val", "test"]:
            for rooms, label in tqdm(zip(self.rooms[split],
                                         self.labels[split])):
                qs = self._room_query_constructor(rooms)
                embedding = self.embedder(qs)
                query_sentence_dict[split].append(qs)
                query_embedding_dict[split].append(embedding)
                label_dict[split].append(label)

        label_tensor_dict = {
            split: torch.tensor(label_dict[split])
            for split in ["train", "val", "test"]
        }
        query_embedding_tensor_dict = {
            split: torch.vstack(query_embedding_dict[split])
            for split in ["train", "val", "test"]
        }
        return query_sentence_dict, query_embedding_tensor_dict, label_tensor_dict

    def _room_query_constructor(self, rooms):
        query_str = "This building contains "
        if len(rooms) > 1:
            for i in rooms[:-1]:
                query_str += self.room_list_pl[i] + ", "
            query_str += "and " + self.room_list_pl[rooms[-1]] + "."
        else:
            query_str += self.room_list_pl[rooms[0]] + "."
        return query_str


if __name__ == "__main__":
    # Goes through bldg_infer.pkl scene graph and creates data for feed-forward
    # method (train / val / test)

    for lm in ["RoBERTa-large", "BERT-large"]:
        data_folder = os.path.join("./building_data/comparison_data",
                                   lm + "_gt")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        num_samples, num_rooms_per_bldg = 1000, 4
        dg = DataGenerator()
        dg.configure_lm(lm)
        dg.extract_data(num_samples, num_rooms_per_bldg)
        qs_dict, qe_tensor_dict, label_tensor_dict = dg.generate_data()

        for split in ["train", "val", "test"]:
            print(qe_tensor_dict[split].shape)
            print(label_tensor_dict[split].shape)

            with open(
                    os.path.join(data_folder,
                                 "query_sentences_" + split + ".pkl"),
                    "wb",
            ) as fp:
                pickle.dump(qs_dict[split], fp)

            # Save labels
            torch.save(label_tensor_dict[split],
                       os.path.join(data_folder, "labels_" + split + ".pt"))

            # Save query embeddings
            torch.save(
                qe_tensor_dict[split],
                os.path.join(data_folder, "query_embeddings_" + split + ".pt"),
            )