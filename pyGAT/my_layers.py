import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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




        # # 4-dim version
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Optimised version
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        # self.W = torch.nn.Parameter(torch.zeros(in_features,out_features))
        # nn.init.xavier_uniform_(self.W, gain=1.414)
        # a1 for i and a2 for j
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # orignial code
        # h = torch.mm(input, self.W)
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # # change to batch level
        # batch_size = len(input)
        # # print(input.shape,self.W.shape)
        # h = torch.matmul(input, self.W)
        # N = h.size()[1] # num of nodes
        # # in the catted vectors, including all the i,j combinations!get tensor: batch_size*N*N*2hidden, too big!
        # # TODO: problem with a_input: should I transpose the two N? or it does not matter for undirected graph?
        # a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)], dim=1).view(batch_size,
        #                                                                                                     N, -1,
        #                                                                                                     2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)

        # Optimised batch model
        # print('self.W',self.W.is_cuda)
        h = self.W(input)
        # h = torch.matmul(input,self.W)
        batch_size,N,_ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        # self.a is a weight vector
        e = self.leakyrelu(middle_result1 + middle_result2)
        # for Avg Pooling setting
        # e = torch.ones(batch_size,N,N).cuda()
        # TODO: here change to soft adjs
        # print(e.dtype,adj.dtype)
        # print(adj)
        # attention = torch.mul(e,adj)
        # print(attention)
        attention = e.masked_fill(adj == 0, -1e9) # reduce a little bit memory
        # print(attention)



        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # print(attention.dtype,h.dtype)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b
#
#
# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)
#
#
# class SpGraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     # good for spare graph!
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(SpGraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)
#
#         self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)
#
#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()
#
#     def forward(self, input, adj):
#         # dv = 'cuda' if input.is_cuda else 'cpu'
#
#         N = input.size()[0]
#         edge = adj.nonzero().t()
#
#         h = torch.mm(input, self.W)
#         # h: N x out
#         assert not torch.isnan(h).any()
#
#         # Self-attention on the nodes - Shared attention mechanism
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E
#
#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
#         # edge_e: E
#
#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)))
#         # e_rowsum: N x 1
#
#         edge_e = self.dropout(edge_e)
#         # edge_e: E
#
#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         assert not torch.isnan(h_prime).any()
#         # h_prime: N x out
#
#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         assert not torch.isnan(h_prime).any()
#
#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
