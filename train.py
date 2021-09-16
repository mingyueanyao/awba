from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import normalize
import scipy.sparse as sp

import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_data, get_neg_table
from graph import load_edgelist_from_csr_matrix
from model import SpGATVAE
from optimizer import walk_regular
from partition import expand




parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--nb_heads', type=int, default=16, help='Number of head attentions.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--size', type=int, default=128, help='Dimension of the output vector.')

parser.add_argument('--number_walks', default=20, type=int, help='Number of random walks to start at each node.')
parser.add_argument('--walk_length', default=10, type=int, help='Length of the random walk started at each node.')
parser.add_argument('--window_size', default=5, type=int, help='Window size of skipgram model.')
parser.add_argument('--negative_num', default=5, type=int, help='Number of negative vertex samples.')
parser.add_argument('--neg_table_size', default=1000000, type=int) # 1e6
parser.add_argument('--neg_sampling_power', default=0.75, type=float)
parser.add_argument('--walk_in_cmu', type=int, default=1)
parser.add_argument('--cmu_walk_p', type=float, default=0.8) 
parser.add_argument('--attention_walk', type=int, default=1)
parser.add_argument('--cluster_frequence', type=int, default=20) 
parser.add_argument('--fitness_alpha', type=float, default=1.2)

parser.add_argument('--net_file', type=str, default='./data/karate_edgelist.txt')
parser.add_argument('--feat_file', type=str, default='')
parser.add_argument('--output_file', type=str, default='karate_embeddings.txt')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def save_node_vec(node_vec, idx_node, args):
    save_path = os.path.join("./output", args.output_file)

    n, dimension = node_vec.shape
    with open(save_path, 'w') as f:
        f.write('{} {}\n'.format(n, dimension))
        for i, node in enumerate(idx_node):
            vec = ' '.join([str(value) for value in node_vec[i]])
            f.write('{} {}\n'.format(str(node), vec))
            
    print('save node_vec to {}'.format(save_path))


def get_vector(feature, adjacency, G, node_idx):
    model = SpGATVAE(nfeat=feature.shape[1], 
                     nhid=args.hidden, 
                     output=args.size, 
                     dropout=args.dropout, 
                     nheads=args.nb_heads, 
                     alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        feature = feature.cuda()
        adjacency = adjacency.cuda()

    feat, adj = Variable(feature), Variable(adjacency)
    output  = None

    walk_loss_values = []
    nodes_in_G = list(G_dw.nodes())
    random.Random().shuffle(nodes_in_G)
    
    cmu_nodes = None
    node_cmu = None

    t_total = time.time()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()

        if args.walk_in_cmu == 0:
            cmu_nodes = None
            node_cmu = None
        model, cur_walk_loss_values = walk_regular(G_dw, nodes_sum, args, 
                                                   model, optimizer, feat, 
                                                   adj, epoch, nodes_in_G, 
                                                   neg_table, node_idx, 
                                                   cmu_nodes, node_cmu)
           
        optimizer.zero_grad()
        z, mu, logvar = model(feat, adj)
        output = mu

        print('\nEpoch: {:04d} '.format(epoch+1), end=' ')
        print('walk_loss: {:<7.4f} '.format(cur_walk_loss_values), end=' ')
        print('time: {:.4f}s'.format(time.time() - t), end=' ')

        if (args.walk_in_cmu == 1) and (epoch % args.cluster_frequence == 0):
            node_vectors = output.cpu().detach().numpy()
            cmu_nodes, node_cmus = expand(node_vectors, G, node_idx, alpha=args.fitness_alpha)

            node_cmu = {node:node_cmus[node][0] for node in node_cmus.keys()}
            cmu_nodes = list(cmu_nodes.values())

    vectors = output.cpu().detach().numpy()
    print("\nOptimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    return vectors


if __name__ == '__main__':
    adj, adj_label, node_idx, features = load_data(args.net_file, args.feat_file)
    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    norm = adj_label.shape[0] * adj_label.shape[0] / float((adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) * 2)
    if args.cuda:
        adj_label = adj_label.cuda()
        pos_weight = pos_weight.cuda()

    G = nx.read_edgelist(args.net_file)
    nodes = sorted(G.nodes(), key=lambda x: (node_idx.get(x)))
    nodes_sum = len(nodes)
    print('nodes sum:', nodes_sum)
    dw_adj = nx.adjacency_matrix(G, nodelist=nodes)
    adj_orig = dw_adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    G_dw = load_edgelist_from_csr_matrix(adj_orig, nodes, undirected=True)
    neg_table = get_neg_table(args.neg_table_size, args.neg_sampling_power, G)

    node_vectors = get_vector(features, adj, G, node_idx)
    save_node_vec(node_vectors, node_idx, args)