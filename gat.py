import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F



# Most of the codes are from https://github.com/Diego999/pyGAT

class GAT(nn.Module):
    def __init__(self, nfeat, nhid,  dropout, alpha, nheads, nclass):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions: ModuleList[GraphAttentionLayer] = ModuleList([])
        for i in range(nheads):
            attention = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            self.attentions.append(attention)
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x




class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat



        # Optimised version
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        h = self.W(input)
        batch_size,N,_ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)



        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'