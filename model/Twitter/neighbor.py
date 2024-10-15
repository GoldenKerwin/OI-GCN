import torch
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F

class Interaction_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Interaction_GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, node_features, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.float() ** 2
        weight_features = self.linear(node_features.float())
        temp = torch.mm(adjacency_matrix, weight_features)
        sum_hadamard = weight_features * temp
        return sum_hadamard

