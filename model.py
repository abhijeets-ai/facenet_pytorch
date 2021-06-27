# __author__ = "Abhijeet Shrivastava"

import torch as th
from torch import nn
from torch.nn import functional as F


class Facenet(nn.Module):
    def __init__(self, head, flatten, dropout, embedding_dim, num_pre_emb_param, add_batch_norm=True):
        '''
        Facenet class models the facenet architecture described in paper: https://arxiv.org/pdf/1503.03832.pdf
        Parameters
        ----------
        head: CNN head to use as feature extractor
        flatten: boolean value to control whether to flatten or use average pooling
        dropout: dropout probability of last layer
        embedding_dim: dimension of output embedding
        num_pre_emb_param: it is equal to num of channel for non-flatten case and for flatten HxWxC of fixed feat map
        add_batch_norm: if batch norm present batch size should be more than 2 of hard sampling
        '''
        super(Facenet, self).__init__()

        self.head = head
        self.flatten = flatten
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.num_last_channel = num_pre_emb_param
        self.add_batch_norm = add_batch_norm
        if not flatten:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(num_pre_emb_param, embedding_dim, bias=False)
        if add_batch_norm:
            self.bn = nn.BatchNorm1d(num_pre_emb_param, eps=0.001, momentum=0.1, affine=True)

    def forward(self, inp: th.tensor) -> th.tensor:
        x = self.head(inp)
        if self.flatten:
            # problem increased computation cost and input should be of fixed size
            x = x.view(x.shape[0], -1)
        else:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
        if self.add_batch_norm:
            x = self.bn(x)
        x = self.dropout(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x
