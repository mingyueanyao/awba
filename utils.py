import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import math




def load_data(net_file, feat_file=''):
    print('Loading {} ...'.format(net_file))
    # build graph
    edges_unordered = np.genfromtxt("{}".format(net_file), dtype=np.dtype(str))
    node_names = []
    if feat_file != '':
        name_features = np.genfromtxt(feat_file, dtype=np.dtype(str))
        features = sp.csr_matrix(name_features[:, 1:], dtype=np.float32)
        node_names = np.array(name_features[:, 0], dtype=np.dtype(str))
    else:
        node_names = set.union(set(np.array(edges_unordered[:, 0], dtype=np.dtype(str))),
                               set(np.array(edges_unordered[:, -1], dtype=np.dtype(str))))

    
    node_idx = {j: i for i, j in enumerate(node_names)} 
    edges = np.array(list(map(node_idx.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(len(node_idx), len(node_idx)), 
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_label = adj.todense() + sp.eye(adj.shape[0])
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_label = torch.FloatTensor(np.array(adj_label))

    if feat_file != '':
        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = adj

    return adj, adj_label, node_idx, features


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_neg_table(neg_table_size, neg_sampling_power, G):
    all_sum = 0.0
    node_degree = list(G.degree())
    for item in node_degree:
        n, d = item
        all_sum += math.pow(d, neg_sampling_power)

    i = node_idx = 0
    cur_sum = por = 0.0
    neg_table = [0] * neg_table_size
    while i < neg_table_size:
        if (i + 1) / neg_table_size > por:
            n, d = node_degree[node_idx]
            cur_sum += math.pow(d, neg_sampling_power)
            por = cur_sum / all_sum
            node_idx += 1
        
        n, d = node_degree[node_idx - 1]
        neg_table[i] = n
        i += 1

    return neg_table