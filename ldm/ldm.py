import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv

class LatentDistanceModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LatentDistanceModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, edge_index):
        # Get embeddings for each node in the edge
        z_i = self.embeddings(edge_index[0])
        z_j = self.embeddings(edge_index[1])

        # Compute Euclidean distance between embeddings
        distance = torch.norm(z_i - z_j, dim=1)
        return distance