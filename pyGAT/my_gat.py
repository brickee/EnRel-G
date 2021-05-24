import torch
import torch.nn as nn
import torch.nn.functional as F
from pyGAT.my_layers import GraphAttentionLayer
from typing import List
from torch.nn import ModuleList


class GAT(nn.Module):
    def __init__(self, nfeat, nhid,  dropout, alpha, nheads, nclass):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        # self.attentions: List[GraphAttentionLayer] = []
        self.attentions: ModuleList[GraphAttentionLayer] = ModuleList([])
        for i in range(nheads):
            attention = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            self.attentions.append(attention)
            self.add_module('attention_{}'.format(i), attention)

        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.attentions2 = [GraphAttentionLayer(nhid*nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(nheads)]
        # for j, attention in enumerate(self.attentions2):
        #     self.add_module('attention_{}'.format(i+j), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions2], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x

# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [SpGraphAttentionLayer(nfeat,
#                                                  nhid,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         # self.out_att = SpGraphAttentionLayer(nhid * nheads,
#         #                                      nclass,
#         #                                      dropout=dropout,
#         #                                      alpha=alpha,
#         #                                      concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         # x = F.elu(self.out_att(x, adj))
#         # return F.log_softmax(x, dim=1)
#         return x
