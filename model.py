import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import SpGraphAttentionLayer

from torch import optim
import random

import numpy as np




class SpGATVAE(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGATVAE, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.mu_att = SpGraphAttentionLayer(nhid * nheads, 
                                            output, 
                                            dropout=dropout, 
                                            alpha=alpha, 
                                            concat=False)

        self.logvar_att = SpGraphAttentionLayer(nhid * nheads, 
                                                output, 
                                                dropout=dropout, 
                                                alpha=alpha, 
                                                concat=False)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, return_attention=False):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        if return_attention:
            mu, attention = self.mu_att(x, adj, return_attention)
        else:
            mu = self.mu_att(x, adj)

        logvar = self.logvar_att(x, adj)
        z = self.reparameterize(mu, logvar)

        # 返回注意力系数矩阵。
        if return_attention:
            return z, mu, logvar, attention
        else:
            return z, mu, logvar