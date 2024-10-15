import torch
import torch.nn as nn

class Interaction_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Interaction_GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, node_features, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.coalesce()
        indices = adjacency_matrix.indices()
        values = adjacency_matrix.values()
        row_indices = indices[0]
        col_indices = indices[1]
        mask = row_indices != col_indices
        new_values = torch.where(mask, values, torch.zeros_like(values))
        new_values = new_values.float() ** 2
        adjacency_matrix_mod = torch.sparse_coo_tensor(
            indices, new_values, adjacency_matrix.size(), device=adjacency_matrix.device
        )
        adjacency_matrix_mod = adjacency_matrix_mod.coalesce()

        weight_features = self.linear(node_features.float())
        temp = torch.sparse.mm(adjacency_matrix_mod, weight_features)
        sum_hadamard = weight_features * temp

        return sum_hadamard