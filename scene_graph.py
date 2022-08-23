import numpy as np
from enum import Enum
from copy import deepcopy
"""
SceneGraph, SceneNode, SceneEdge, NodeType classes
"""


class SceneGraph:
    """
    SceneGraph class
      nodes      a list of SceneNode
      edge_dict  dictionary of SceneEdges {(start_idx, end_idx):
            a list of SceneEdge between start and end nodes where start_idx and end_idx are
            the node's respective indices in the nodes list } # TODO
    """

    def __init__(self):
        self.__nodes = []
        self.__edge_dict = dict()
        self.__hierarchy = [NodeType.building, NodeType.room, NodeType.object]

    def num_nodes(self, node_type=None):
        if node_type is None:
            return len(self.__nodes)
        else:
            return sum(node.node_type == node_type for node in self.__nodes)

    def num_edges(self):
        return sum([len(v) for v in self.__edge_dict.values()])

    def get_node(self, node_idx):
        return self.__nodes[node_idx]

    def get_node_by_id_type(self, node_id, node_type):
        filtered_nodes = list(
            filter(lambda x: x.node_id == node_id and x.node_type == node_type,
                   self.__nodes))
        if len(filtered_nodes) == 0:
            return None
        elif len(filtered_nodes) == 1:
            return filtered_nodes[0]
        else:
            raise RuntimeError(
                'get_node_by_id_type() found more than one nodes.')

    def get_edge(self, start_idx, end_idx, rel):
        return next((edge for edge in self.__edge_dict[(start_idx, end_idx)]
                     if edge.rel == rel), None)

    def get_edge_relationships(self, start_idx, end_idx):
        return [edge.rel for edge in self.__edge_dict[(start_idx, end_idx)]]

    def get_edges(self, start_idx, end_idx):
        return self.__edge_dict[(start_idx, end_idx)]

    def get_nodes_copy(self):
        return deepcopy(self.__nodes)

    def get_edge_dict_copy(self):
        return deepcopy(self.__edge_dict)

    def get_hierarchy_copy(self):
        return deepcopy(self.__hierarchy)

    def set_hierarchy(self, new_hierarchy):
        for layer in new_hierarchy:
            assert isinstance(layer, NodeType)
        self.__hierarchy = new_hierarchy

    def get_adjacent_node_indices(self, node_idx):
        out_indices = [
            idx_pair[1] for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[0] == node_idx
        ]
        in_indices = [
            idx_pair[0] for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[1] == node_idx
        ]
        return out_indices, in_indices

    def find_parent_idx(self, scene_node):
        if scene_node.node_type == NodeType.building:
            return None

        node_idx = self.__nodes.index(scene_node)
        expected_type_idx = self.__hierarchy.index(scene_node.node_type) - 1
        expected_type = self.__hierarchy[expected_type_idx]

        parent_indices = [
            idx_pair[1] for idx_pair in list(self.__edge_dict.keys())
            if idx_pair[0] == node_idx
            and self.__nodes[idx_pair[1]].node_type == expected_type
        ]
        if len(parent_indices) == 0:
            return None
        elif len(parent_indices) == 1:
            return parent_indices[0]
        else:
            print('Warning: {} has more than one parent.'.format(
                self.__nodes[node_idx]))
            return parent_indices[0]

    def get_relationship_set(self):
        return set(scene_edge.rel
                   for scene_edge in sum(self.__edge_dict.values(), []))

    def add_node(self, new_node):
        assert isinstance(new_node, SceneNode)
        if new_node not in self.__nodes:
            self.__nodes.append(new_node)

    def add_edge(self, new_edge):
        assert isinstance(new_edge, SceneEdge)
        if new_edge.weight == 0:  # do not update when weight is 0
            return

        # update self.__nodes
        try:
            start_idx = self.__nodes.index(new_edge.start)
        except ValueError:
            start_idx = len(self.__nodes)
            self.__nodes.append(new_edge.start)  # make shallow copy
        try:
            end_idx = self.__nodes.index(new_edge.end)
        except ValueError:
            end_idx = len(self.__nodes)
            self.__nodes.append(new_edge.end)  # make shallow copy

        # update self.__edge_dict
        # TODO: delete print after debugging
        if (start_idx, end_idx) in self.__edge_dict.keys():
            try:
                edge_idx = self.__edge_dict[(start_idx,
                                             end_idx)].index(new_edge)
                self.__edge_dict[(start_idx, end_idx)][edge_idx] = new_edge
                print("Update weight of edge {}".format(new_edge))
            except ValueError:
                print(
                    "Additional relationship ({}) between scene node {} and {}"
                    .format(new_edge.rel, new_edge.start, new_edge.end))
                self.__edge_dict[(start_idx, end_idx)].append(new_edge)
        else:
            self.__edge_dict[(start_idx, end_idx)] = [new_edge]

        if new_edge.start.node_type != new_edge.end.node_type and new_edge.rel == "AtLocation":
            if new_edge.start.node_type not in self.__hierarchy:
                parent_layer = new_edge.end.node_type
                idx_parent_layer = self.__hierarchy.index(parent_layer)
                self.__hierarchy = self.__hierarchy[:idx_parent_layer+1] + [new_edge.start.node_type] \
                                   + self.__hierarchy[idx_parent_layer+1:]
                print("hierarchy of scene graph updated to", self.__hierarchy)
            elif new_edge.end.node_type not in self.__hierarchy:
                child_layer = new_edge.start.node_type
                idx_child_layer = self.__hierarchy.index(child_layer)
                self.__hierarchy = self.__hierarchy[:idx_child_layer] + [new_edge.end.node_type] \
                                   + self.__hierarchy[idx_child_layer:]
                print("hierarchy of scene graph updated to", self.__hierarchy)

    def generate_adjacency_matrix(self):
        nr_nodes = len(self.__nodes)
        adjacency_matrix = np.zeros((nr_nodes, nr_nodes), dtype=bool)

        # A[i, j] = True when there is an edge from the i-th node to the j-th node in self.nodes
        start_indices = [
            edge_indices[0] for edge_indices in self.__edge_dict.keys()
        ]
        end_indices = [
            edge_indices[1] for edge_indices in self.__edge_dict.keys()
        ]
        adjacency_matrix[start_indices, end_indices] = True
        return adjacency_matrix

    def num_correct_labels(self, scene_graph_ref):
        correct_labels = 0
        for i, node in enumerate(self.__nodes):
            if node.semantic_label == scene_graph_ref.get_node(
                    i).semantic_label:
                correct_labels += 1
        return correct_labels


class SceneEdge:
    """
    SceneEdge class
      start      SceneNode
      rel        string or None for unknown relationship (ConceptNet and VG relationships or None)
      end        SceneNode
      weight     float
    """

    def __init__(self, start, rel, end, weight=1.0):
        assert isinstance(start, SceneNode)
        assert isinstance(end, SceneNode)
        self.start = start
        self.rel = rel
        self.end = end
        self.weight = weight

    def __str__(self):
        return "{0} - {1} - {2}".format(self.start, self.rel, self.end)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # start, rel, end all have to be the same, but weight does not matter
        if not isinstance(other, SceneEdge):
            # don't attempt to compare against unrelated types
            return NotImplemented

        if not (self.start == other.start and self.rel == other.rel
                and self.end == other.end):
            return False
        elif self.weight != other.weight:  # TODO: remove after debug
            print("same scene edge with different weight")
            return True
        else:
            return True


class NodeType(Enum):
    human = 0  # not used right now
    object = 1
    room = 2
    building = 3
    place = 4  # not used by CRF class


class SceneNode:
    """
    SceneNode class
      node_id             int (unique for each node in the same graph)
      node_type           SceneNodeType (objects, rooms, etc. or layer)
      semantic_label      string
      centroid            1d numpy array
      size                1d numpy array or None # TODO: on hold
      possible_labels     a list of strings or None
      label_weights       1d numpy array of weights corresponding to semantic_label in possible_labels
    """

    def __init__(self,
                 node_id,
                 node_type,
                 centroid,
                 size=None,
                 semantic_label=None,
                 possible_labels=None,
                 label_weights=None):
        assert isinstance(node_type, NodeType)

        self.node_id = node_id
        self.node_type = node_type
        self.semantic_label = semantic_label
        self.centroid = np.array(centroid)
        self.size = None if size is None else np.array(size)
        self.possible_labels = possible_labels
        self.label_weights = np.array(label_weights)

    def __str__(self):
        semantic_label = self.semantic_label if self.semantic_label is not None else 'None'
        return '%s (%d)' % (semantic_label, self.node_id)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.node_id, self.node_type, self.semantic_label))

    def __eq__(self, other):
        # compare id, node_type and semantic_label
        if self.node_id == other.node_id and self.node_type == other.node_type and \
                self.semantic_label == other.semantic_label:
            return True
        elif self.node_id == other.node_id and self.node_type == other.node_type:  # Todo: for debugging
            print("Same node id and type but different semantic label")
            return False
        else:
            return False
