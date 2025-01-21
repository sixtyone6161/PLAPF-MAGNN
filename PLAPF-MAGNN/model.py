from torch_geometric.utils import softmax
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from SGAT_SAGE import NSAGE, ESGAT

class PAIGN(nn.Module):
    def __init__(self, node_features_dim, hidden_channels, num_classes):
        super(PAIGN, self).__init__()


        self.nsage_conv1 = NSAGE(node_features_dim, hidden_channels)
        self.nsage_conv2 = NSAGE(hidden_channels, hidden_channels * 2)
        self.nsage_conv3 = NSAGE(hidden_channels * 2, hidden_channels * 2)


        self.esgat_conv1 = ESGAT(node_features_dim, hidden_channels)
        self.esgat_conv2 = ESGAT(hidden_channels, hidden_channels * 2)
        self.esgat_conv3 = ESGAT(hidden_channels * 2, hidden_channels * 2)


        self.reduce_dim1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.reduce_dim2 = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.reduce_dim3 = nn.Linear(hidden_channels * 4, hidden_channels * 2)


        self.residual1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.residual2 = nn.Linear(hidden_channels * 2, hidden_channels * 2)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2)



        self.lin = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x, edge_index, batch):

        x_sage1 = self.nsage_conv1(x, edge_index)
        x_esgat_1 = self.esgat_conv1(x, edge_index)
        x1 = torch.cat([x_sage1, x_esgat_1], dim=-1)
        x1 = self.reduce_dim1(x1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)


        x_sage2 = self.nsage_conv2(x1, edge_index)
        x_esgat_2 = self.esgat_conv2(x1, edge_index)
        x2 = torch.cat([x_sage2, x_esgat_2], dim=-1)
        x2 = self.reduce_dim2(x2)

        x2 = x2 + 1e-7 * F.relu(self.residual1(x1))

        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x_sage3 = self.nsage_conv3(x2, edge_index)
        x_esgat_3 = self.esgat_conv3(x2, edge_index)
        x3 = torch.cat([x_sage3, x_esgat_3], dim=-1)
        x3 = self.reduce_dim3(x3)

        x3 = x3 + 1e-8 * F.relu(self.residual2(x2))

        x3 = self.bn3(x3)
        x3 = F.relu(x3)


        x3 = global_add_pool(x3, batch)


        x3 = F.dropout(x3, p=0.5, training=self.training)


        x3 = self.lin(x3)

        return x3







