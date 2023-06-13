from scene_graph import NodeType
import pickle


class Matterport3dDataset:

    def __init__(self, dataset_file_path):
        with open(dataset_file_path, 'rb') as input_file:
            input_data = pickle.load(input_file)
        dataset_list = input_data[0]
        labels_dict = input_data[1]
        train_val_test_split = input_data[2]
        self.num_node_features = dataset_list[0].x.shape[1]
        """ self.num_classes = [
            len(labels_dict[NodeType.building]),
            len(labels_dict[NodeType.room]),
            len(labels_dict[NodeType.object])
        ] """

        self.num_classes = [
            len(labels_dict[key]) for key in labels_dict.keys()
        ]

        self.labels_dict = labels_dict
        self.train_val_test_split = train_val_test_split
        self.name = 'Matterport3d_full_graph'
        self.__dataset = dataset_list

        if len(input_data) > 3:
            self.semantic_embedding_dict = input_data[3]
        else:
            self.semantic_embedding_dict = None

    def __getitem__(self, item):
        return self.__dataset[item]

    def __len__(self):
        return len(self.__dataset)

    def get_training_set(self):
        return [self.__dataset[i] for i in self.train_val_test_split[0]]

    def get_validation_set(self):
        return [self.__dataset[i] for i in self.train_val_test_split[1]]

    def get_test_set(self):
        return [self.__dataset[i] for i in self.train_val_test_split[2]]
