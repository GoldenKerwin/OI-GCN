import torch
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F


class Interaction_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Interaction_GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, node_features, adjacency_matrix, sibling_adj):
        M = sibling_adj.float()
        weight_features = self.linear(node_features.float())
        temp = torch.mm(M, weight_features)
        sum_hadamard = weight_features * temp
        sum_hadamard = torch.mm(adjacency_matrix, sum_hadamard)

        return sum_hadamard



