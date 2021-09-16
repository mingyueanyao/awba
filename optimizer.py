from __future__ import division
from __future__ import print_function

import os, sys
import random
import numpy as np
import torch
import torch.nn.functional as F

from graph import build_deepwalk_corpus_iter




def walk_regular(G, n_nodes, args, model, optimizer, feat, adj, epoch, 
                 nodes_in_G, neg_table, node_idx, cmu_nodes, node_cmu):
    att_mtx = None
    if args.attention_walk == 1:
        z, mu, logvar, att_mtx = model(feat, adj, args.attention_walk)

    chunks = len(nodes_in_G) // args.number_walks
    if chunks == 0:
        chunks += 1

    walks = build_deepwalk_corpus_iter(G, num_paths=args.number_walks,
                                       path_length=args.walk_length, alpha=0,
                                       rand=random.Random(args.seed), chunk=epoch % chunks,
                                       nodes=nodes_in_G, cmu_nodes=cmu_nodes, 
                                       node_cmu=node_cmu, att_mtx=att_mtx, 
                                       node_idx=node_idx, cmu_walk_p=args.cmu_walk_p)

    return update_use_walks(walks, args, n_nodes, optimizer, model, feat, adj, neg_table, node_idx)


def update_use_walks(walks, args, n_nodes, optimizer, model, feat, adj, neg_table, node_idx): 
    cnt = 0.
    all_walk_loss = 0.
    
    mask_mtx = np.zeros((n_nodes, n_nodes), dtype=float)
    device = torch.device('cuda' if args.cuda else 'cpu')

    optimizer.zero_grad()
    output = get_encoder_output(model, feat, adj, args)
    for walk in walks:
        for center_node_pos in range(len(walk)):
            src_node, tgt_nodes, neg_nodes = get_node_pairs(center_node_pos,
                                                            walk, node_idx, 
                                                            args, neg_table)

            if len(tgt_nodes) != 0:
                update_mask_mtx(mask_mtx, src_node, tgt_nodes, neg_nodes)

            cnt = cnt + 1

    mask_mtx = torch.FloatTensor(mask_mtx).to(device)
    z, mu, logvar = model(feat, adj)
    all_walk_loss_ts = walk_loss_all_in_one(mu, logvar, mask_mtx, device)

    all_walk_loss_ts.backward()
    all_walk_loss = all_walk_loss_ts.item() 
        
    optimizer.step()

    return model, all_walk_loss/cnt


def walk_loss_all_in_one(mu, logvar, mask_mtx, device):
    raw_score_mtx = torch.matmul(mu, logvar.t())
    raw_score_mtx = torch.where(mask_mtx < 0, -raw_score_mtx, raw_score_mtx)
    raw_score_mtx = F.logsigmoid(raw_score_mtx)
    mask_mtx = torch.where(mask_mtx < 0, -mask_mtx, mask_mtx)
    score_mtx = raw_score_mtx.mul(mask_mtx)

    return  -1 * torch.sum(score_mtx)


def update_mask_mtx(mask_mtx, src, tgts, negs):
    ratio = 1.0 / len(tgts)

    for t in tgts:
        mask_mtx[src][t] += ratio

    for n in negs:
        mask_mtx[src][n] -= ratio

    return mask_mtx

    
def get_encoder_output(model, feat, adj, args):
    try:
        z, mu, logvar = model(feat, adj)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception
    
    output = mu
    return output


def get_node_pairs(center_node_pos, walk, node_idx, args, neg_table):
    snode_idx = node_idx[walk[center_node_pos]]
    curr_pair = (snode_idx, [])
    walk_len = len(walk)

    # for each window position
    for w in range(-args.window_size, args.window_size + 1):
        context_node_pos = center_node_pos + w
        # make soure not jump out sentence
        if context_node_pos < 0 or context_node_pos >= walk_len or center_node_pos == context_node_pos:
            continue
        cnode_idx = node_idx[walk[context_node_pos]]
        curr_pair[1].append(cnode_idx)

    neg_nodes = []
    neg_sample_num = 0
    pos_nodes = set(curr_pair[1])
    pos_sample_num = len(curr_pair[1])

    while neg_sample_num < pos_sample_num:
        rand_node = random.choice(neg_table)
        rnode_idx = node_idx[rand_node]

        if rnode_idx != snode_idx and (rnode_idx not in pos_nodes):
            neg_nodes.append(rnode_idx)
            neg_sample_num += 1

    src_node = torch.from_numpy(np.array([curr_pair[0]])).long()
    tgt_nodes = torch.from_numpy(np.array(curr_pair[1])).long()
    neg_nodes = torch.from_numpy(np.array(neg_nodes)).long()

    return src_node, tgt_nodes, neg_nodes