import torch
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch_geometric.nn import MultiAggregation,global_add_pool,GATConv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch import Tensor, nn

class NSAGE(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)


        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = nn.Linear(aggr_out_channels + 3, out_channels)

        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])


        neighbors = self.propagate(edge_index, x=x, size=size)
        mean_neighbors = neighbors.mean(dim=1, keepdim=True)
        max_neighbors = neighbors.max(dim=1, keepdim=True)[0]
        std_neighbors = neighbors.std(dim=1, keepdim=True)


        neighbors = torch.cat([neighbors, mean_neighbors, max_neighbors, std_neighbors], dim=-1)


        out = self.lin_l(neighbors)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')




class ESGAT(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(ESGAT, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, negative_slope=negative_slope)

    def forward(self, x, edge_index):

        x, (edge_index, attention) = self.gat_conv(x, edge_index, return_attention_weights=True)


        degree = torch.bincount(edge_index[1], minlength=x.size(0))
        degree = degree[edge_index[1]]


        scaling_score = 1.0 / degree.float()
        scaling_score = torch.where(degree > 1, scaling_score, torch.ones_like(scaling_score))


        attention = attention * scaling_score.view(-1, 1)
        attention = softmax(attention, edge_index[0], num_nodes=x.size(0))


        out = torch.zeros_like(x)
        out = torch.scatter_add(out, 0, edge_index[1].unsqueeze(-1).expand(-1, x.size(1)), attention * x[edge_index[0]])

        return out

