import torch
import torch.nn as nn

# An equivalent sparse matrix algorithm is used instead of a dense matrix
class Interaction_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Interaction_GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def scale_sparse_matrix_rows(self, sparse_matrix, scale_vector):

        scale_vector = scale_vector.squeeze()
        diag_indices = torch.arange(scale_vector.size(0), device=scale_vector.device)
        diag_indices = torch.stack([diag_indices, diag_indices], dim=0)
        diag_values = scale_vector
        diag_matrix = torch.sparse_coo_tensor(diag_indices, diag_values, sparse_matrix.size())
        scaled_sparse_matrix = torch.sparse.mm(diag_matrix, sparse_matrix)
        return scaled_sparse_matrix

    def forward(self, node_features, adjacency_matrix, degree):
        adjacency_matrix_unnormalized = self.scale_sparse_matrix_rows(adjacency_matrix, degree)
        

        degree_unsqueezed = degree.unsqueeze(1)
        degree_brother = torch.sparse.mm(adjacency_matrix_unnormalized, degree_unsqueezed) - degree_unsqueezed  # [N, 1]
        degree_brother = degree_brother.squeeze(1)
        degree_brother = torch.where(degree_brother == 0, torch.tensor(1.0, device=degree_brother.device), degree_brother)

        adjacency_matrix_brother_normalized = self.scale_sparse_matrix_rows(
            adjacency_matrix_unnormalized, 1.0 / degree_brother
        )
        
        weight_features = self.linear(node_features.float())
        temp1 = torch.sparse.mm(adjacency_matrix_unnormalized, weight_features)
        temp2 = torch.sparse.mm(adjacency_matrix_brother_normalized, temp1)

        degree = degree.unsqueeze(1)
        degree = torch.where(degree == 0, torch.tensor(1.0, device=degree.device), degree)
        temp2 = temp2 / degree
        temp = temp2 - weight_features 
        sum_hadamard = weight_features * temp
        sum_hadamard = torch.sparse.mm(adjacency_matrix, sum_hadamard)
        return sum_hadamard



