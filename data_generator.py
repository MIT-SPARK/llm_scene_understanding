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

        self.dataset = Matterport3dDataset("./mp_data/" + label_set +
                                           "_matterport3d_w_edge_502030.pkl")
        labels, pl_labels = create_label_lists(self.dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        self.excluded_rooms = ["None", "yard", "porch", "balcony"]

        if self.verbose:
            print("Using device:", self.device)

        # create data loader
        # self.dataloader = DataLoader(dataset, batch_size=82)
        if use_gt_cooccurrencies:
            path_to_cooccurrencies = ("./cooccurrency_matrices/" + label_set +
                                      "_gt" + "/room_object.npy")
        else:
            path_to_cooccurrencies = ("./cooccurrency_matrices/" + label_set +
                                      "_gpt_j" + "/room_object.npy")

        self.object_norm_inv_perplexity = compute_object_norm_inv_ppl(
            path_to_cooccurrencies,
            use_gt_cooccurrencies,
        ).to(self.device)

        self.lm = None
        self.lm_model = None
        self.tokenizer = None
        self.embedder = None

        if default_lm is not None:
            self.configure_lm(default_lm)

        self.max_num_obj = None
        self.objects = None
        self.labels = None
        # self.object_counts = None

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

    def reset_data(self):
        self.objects = []
        self.labels = []
        self.all_objs = []

    def extract_data(self, max_num_obj, split=""):
        """
        Extracts and saves the most interesting objects from each room.

        TODO: Finish docstring
        """
        self.max_num_obj = max_num_obj
        self.reset_data()

        if split == "train":
            dataloader = DataLoader(self.dataset.get_training_set(),
                                    batch_size=82)
        elif split == "val":
            dataloader = DataLoader(self.dataset.get_validation_set(),
                                    batch_size=82)
        elif split == "test":
            dataloader = DataLoader(self.dataset.get_test_set(), batch_size=82)
        else:
            dataloader = DataLoader(self.dataset, batch_size=82)

        batch = next(iter(dataloader))
        label = (
            batch.y[batch.building_mask],
            batch.y[batch.room_mask],
            batch.y[batch.object_mask],
        )
        y_object = F.one_hot(label[-1],
                             len(self.object_list)).type(torch.LongTensor)
        category_index_map = get_category_index_map(batch)
        object_room_edge_index = batch.object_room_edge_index

        for i in tqdm(range(len(label[1]))):  # range(len(label[1])):
            ground_truth_room = label[1][i]

            mask = category_index_map[object_room_edge_index[1]] == i
            neighbor_dists = y_object[category_index_map[
                object_room_edge_index[0][mask]]]
            neighbor_dists = neighbor_dists.to(self.device)
            all_objs = torch.sum(neighbor_dists, dim=0) > 0

            room_label = self.room_list[ground_truth_room]
            if room_label in self.excluded_rooms or len(neighbor_dists) == 0:
                continue

            scores = all_objs * self.object_norm_inv_perplexity

            objs = torch.topk(scores,
                              max(min((all_objs > 0).sum(), max_num_obj),
                                  1)).indices
            all_obj_names = [self.object_list[i] for i in all_objs.nonzero()]

            self.objects.append(objs)
            self.labels.append(ground_truth_room)
            self.all_objs.append(all_obj_names)

    def generate_data(self, k, num_objs, all_permutations=True, skip_rms=True):
        """
        Constructs query string using selected number of objects

        Args:
            k: int <= num_objs number of objects to include in
                query string
            num_objs: int <= self.max_num_objs number of objects
                to choose k out of when generating query strings. Prioritizes
                most semantically interesting objects.

        Returns:
            Tuple of (list of strs, torch.tensor, torch.tensor, torch.tensor).
                Respectively:
                1) list of query sentences of length
                    (# rooms) * (num_obs P k)
                2) tensor of int room labels corresponding to above list
                3) tensor of sentence embeddings corresponding to above list
                4) tensor of sentence embeddings corresponding to room label string
        """
        query_sentence_list = []
        label_list = []
        query_embedding_list = []
        room_embedding_list = []
        all_objs_list = []

        for objs, label, all_objs in tqdm(
                zip(self.objects, self.labels, self.all_objs)):
            if skip_rms:
                if len(objs) < num_objs:
                    continue
            else:
                if len(objs) == 0:
                    continue

            k_room = min(len(objs), k)
            n = min(len(objs), num_objs)

            np_objs = objs[:n].cpu().numpy()
            np_label = label.cpu().numpy()
            if all_permutations:
                for np_objs_p in multiset_permutations(np_objs, k_room):
                    objs_p = torch.tensor(np_objs_p)
                    query_str = self._object_query_constructor(objs_p)
                    room_str = self._room_str_constructor(np_label)

                    query_embedding = self.embedder(query_str)
                    room_embedding = self.embedder(room_str)

                    query_sentence_list.append(query_str)
                    label_list.append(label)
                    query_embedding_list.append(query_embedding)
                    room_embedding_list.append(room_embedding)
                    all_objs_list.append(all_objs)
            else:
                objs_p = torch.tensor(np_objs)
                query_str = self._object_query_constructor(objs_p)
                room_str = self._room_str_constructor(np_label)

                query_embedding = self.embedder(query_str)
                room_embedding = self.embedder(room_str)

                query_sentence_list.append(query_str)
                label_list.append(label)
                query_embedding_list.append(query_embedding)
                room_embedding_list.append(room_embedding)
                all_objs_list.append(all_objs)
        return (
            query_sentence_list,
            all_objs_list,
            torch.tensor(label_list),
            torch.vstack(query_embedding_list),
            torch.vstack(room_embedding_list),
        )

    def _object_query_constructor(self, objects):
        """
        Construct a query string based on a list of objects

        Args:
            objects: torch.tensor of object indices contained in a room

        Returns:
            str query describing the room, eg "This is a room containing
                toilets and sinks."
        """
        assert len(objects) > 0
        query_str = "This room contains "
        names = []
        for ob in objects:
            names.append(self.object_list_pl[ob])
        if len(names) == 1:
            query_str += names[0]
        elif len(names) == 2:
            query_str += names[0] + " and " + names[1]
        else:
            for name in names[:-1]:
                query_str += name + ", "
            query_str += "and " + names[-1]
        query_str += "."
        return query_str

    def _room_str_constructor(self, room):
        room_str = self.room_list[room]
        if room_str != "utility room" and room_str[0] in "aeiou":
            return "An " + room_str + "."
        else:
            return "A " + room_str + "."

    def data_split_generator(self, data_generation_params, k_test):
        max_n = np.max([i[1] for i in data_generation_params])

        split_dict = {}

        # Train
        dg.extract_data(max_n, split="train")
        TEMP = {}
        count = 0
        for k, total in data_generation_params:
            suffix = "train_k" + str(k) + "_total" + str(total)
            sentences, all_objs_list, labels, query_embeddings, room_embeddings = dg.generate_data(
                k, total)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, all_objs_list, labels, query_embeddings,
                room_embeddings
            ]
        split_dict["train"] = TEMP
        print(count, "train sentences")

        # Val
        dg.extract_data(max_n, split="val")
        TEMP = {}
        count = 0
        for k, total in data_generation_params:
            suffix = "val_k" + str(k) + "_total" + str(total)
            sentences, all_objs_list, labels, query_embeddings, room_embeddings = dg.generate_data(
                k, total)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, all_objs_list, labels, query_embeddings,
                room_embeddings
            ]
        split_dict["val"] = TEMP
        print(count, "val sentences")

        # Test
        if k_test > 0:
            dg.extract_data(max_n, split="test")
            TEMP = {}
            count = 0
            suffix = "test_k" + str(k_test)
            sentences, all_objs_list, labels, query_embeddings, room_embeddings = dg.generate_data(
                k_test, k_test, all_permutations=False, skip_rms=False)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, all_objs_list, labels, query_embeddings,
                room_embeddings
            ]
            split_dict["test"] = TEMP
            print(count, "test sentences")
        return split_dict


if __name__ == "__main__":
    for lm in ["RoBERTa-large", "BERT-large"]:
        for label_set in ["nyuClass", "mpcat40"]:
            for use_gt in [True, False]:
                data_folder = os.path.join(
                    "./data/",
                    lm + "_" + label_set + "_useGT_" + str(use_gt) + "_502030")
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                for split in ["train", "val", "test"]:
                    if not os.path.exists(os.path.join(data_folder, split)):
                        os.makedirs(os.path.join(data_folder, split))

                dg = DataGenerator(verbose=True,
                                   label_set=label_set,
                                   use_gt_cooccurrencies=use_gt)
                dg.configure_lm(lm)

                data_generation_params = [(1, 1), (2, 2), (3, 3), (1, 2),
                                          (2, 3), (3, 4)]
                k_test = 3

                split_dict = dg.data_split_generator(data_generation_params,
                                                     k_test)
                # Save
                splits = ["train", "val", "test"
                          ] if k_test > 0 else ["train", "val"]
                for split in splits:
                    for suffix in split_dict[split]:
                        sentences, all_objs_list, labels, query_embeddings, room_embeddings = split_dict[
                            split][suffix]

                        # Save query sentences
                        with open(
                                os.path.join(
                                    data_folder, split,
                                    "query_sentences_" + suffix + ".pkl"),
                                "wb",
                        ) as fp:
                            pickle.dump(sentences, fp)

                        # Save list of strings of all objects
                        with open(
                                os.path.join(data_folder, split,
                                             "all_objs_" + suffix + ".pkl"),
                                "wb",
                        ) as fp:
                            pickle.dump(all_objs_list, fp)

                        # Save labels
                        torch.save(
                            labels,
                            os.path.join(data_folder, split,
                                         "labels_" + suffix + ".pt"))

                        # Save query embeddings
                        torch.save(
                            query_embeddings,
                            os.path.join(data_folder, split,
                                         "query_embeddings_" + suffix + ".pt"),
                        )

                        # Save room embeddings
                        torch.save(
                            room_embeddings,
                            os.path.join(data_folder, split,
                                         "room_embeddings_" + suffix + ".pt"),
                        )