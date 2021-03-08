import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, infeat, bsize, topology, n_class, dropout):
        super(GCN, self).__init__()

        self.num_layers = len(topology)
        self.layers = nn.ModuleDict(
            {'gc{}'.format(i): GraphConvolution(infeat, topology, bsize, i, n_class) for i in range(self.num_layers)})
        self.outlayer = GraphConvolution(infeat, topology, bsize, self.num_layers, n_class)
        self.dropout = dropout

    def forward(self, x, adj, ls=False):
        for i in range(self.num_layers):
            x = self.layers['gc' + str(i)](x, adj)
            x = F.relu(x)
            if i == 0:
                x = F.dropout(x, self.dropout, training=self.training)
        if ls:
            pred = x
        else:
            x = self.outlayer(x, adj)
            pred = F.log_softmax(x, dim=1)
        return pred
