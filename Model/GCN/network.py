from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
import torch


class MolecularGCN(torch.nn.Module):
    def __init__(self, dim, n_conv_hidden, n_mlp_hidden, dropout):
        super(MolecularGCN, self).__init__()
        self.n_features = 75 #  This is the mol2graph.py-specific value
        self.n_conv_hidden = n_conv_hidden
        self.n_mlp_hidden = n_mlp_hidden
        self.dim = dim
        self.dropout = dropout
        self.graphconv1 = GCNConv(self.n_features, self.dim, cached=False)
        self.bn1 = BatchNorm1d(self.dim)
        self.graphconv_hidden = ModuleList(
            [GCNConv(self.dim, self.dim, cached=False) for _ in range(self.n_conv_hidden)]
        )
        self.bn_conv = ModuleList(
            [BatchNorm1d(self.dim) for _ in range(self.n_conv_hidden)]
        )
        self.mlp_hidden =  ModuleList(
            [Linear(self.dim, self.dim) for _ in range(self.n_mlp_hidden)]
        )
        self.bn_mlp = ModuleList(
            [BatchNorm1d(self.dim) for _ in range(self.n_mlp_hidden)]
        )
        self.mlp_out = Linear(self.dim, 1000) # classification of 1000 templates.

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = F.relu(self.graphconv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        for graphconv, bn_conv in zip(self.graphconv_hidden, self.bn_conv):
            x = graphconv(x, edge_index, edge_weight)
            x = bn_conv(x)
        x = global_add_pool(x, batch)
        for fc_mlp, bn_mlp in zip(self.mlp_hidden, self.bn_mlp):
            x = F.relu(fc_mlp(x))
            x = bn_mlp(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.mlp_out(x), dim=-1)
        return x