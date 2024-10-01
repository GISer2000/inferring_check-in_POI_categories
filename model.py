import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_RES(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers=5):
        super(GCN_RES, self).__init__()
        # 网络层数
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 输入层
        self.convs.append(GCNConv(dim_in, dim_h))
        self.norms.append(nn.LayerNorm(dim_h))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim_h, dim_h))
            self.norms.append(nn.LayerNorm(dim_h))
        
        # 输出层
        self.convs.append(GCNConv(dim_h, dim_out))
        self.norms.append(nn.LayerNorm(dim_out))

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            if i == 0:
                h = self.convs[i](x, edge_index, edge_weight)
                h = self.norms[i](h)
                res = F.relu(h) + x
                h = F.dropout(res, p=0.6, training=self.training)
                
            elif i != (self.num_layers - 1):
                h = self.convs[i](h, edge_index, edge_weight)
                h = self.norms[i](h)
                res = F.relu(h) + res
                h = F.dropout(res, p=0.6, training=self.training)
                
            else:
                h = self.convs[i](h, edge_index, edge_weight)
                h = self.norms[i](h)

        return h