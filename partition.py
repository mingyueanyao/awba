import math
import numpy as np
import networkx as nx




def cal_dist(n1, n2, vec, node_idx):
    idx1, idx2 = node_idx[n1], node_idx[n2]
    x = np.inner(vec[idx1], vec[idx2])
    return -np.log(1.0 / (1 + np.exp(-x)))


def nearest_neighbor(node, G, vec, node_idx):
    nearest = -1
    d, dmin = 0, float('inf')
    for n in nx.neighbors(G, node):
        d = cal_dist(n, node, vec, node_idx)
        if d < dmin:
            dmin = d
            nearest = n
    
    return nearest


def cal_cmu_info(node, cmu, cmu_info, node_cmus, G, vec, node_idx):
    in_dist, out_dist = cmu_info.get(cmu)
    for n in nx.neighbors(G, node):
        n_cmus = node_cmus.get(n, [-1])
        dist = cal_dist(n, node, vec, node_idx)
        if cmu in n_cmus:
            in_dist = in_dist + dist + dist
            out_dist = out_dist - dist
        else:
            out_dist = out_dist + dist

    return in_dist, out_dist


def update_cmu_info(cmu_info, cmu, new_node, G, node_cmus, vec, node_idx):
    in_dist, out_dist = cal_cmu_info(new_node, cmu, cmu_info, 
                                     node_cmus, G, vec, node_idx)  
        
    cmu_info[cmu][0] = in_dist
    cmu_info[cmu][1] = out_dist


def cal_cmu_edges(cmu_edges, cmu, new_node, G, node_cmus):
    in_edges, out_edges = cmu_edges.get(cmu)
    for n in nx.neighbors(G, new_node):
        n_cmus = node_cmus.get(n, [-1])
        if cmu in n_cmus: 
            in_edges = in_edges + 1 + 1
            out_edges = out_edges - 1
        else:
            out_edges = out_edges + 1

    return in_edges, out_edges


def update_cmu_edges(cmu_edges, cmu, new_node, G, node_cmus):
    in_edges, out_edges = cal_cmu_edges(cmu_edges, cmu,
                                        new_node, G, node_cmus)

    cmu_edges[cmu][0] = in_edges
    cmu_edges[cmu][1] = out_edges


def merge(n, cmu, node_cmus, cmu_nodes, cmu_info, 
          cmu_edges, G, vec, node_idx):
    n_cmus = node_cmus.get(n, [-1])
    if n_cmus[0] == -1:
        node_cmus[n] = [cmu]
    else:
        node_cmus[n].append(cmu)

    cmu_nodes[cmu].append(n)
    update_cmu_info(cmu_info, cmu, n, G, 
                    node_cmus, vec, node_idx)

    update_cmu_edges(cmu_edges, cmu, n, G, node_cmus)


def merge_neighbor(seed, cmu, G, vec, node_idx, node_cmus, cmu_nodes, 
                   cmu_info, cmu_edges):
    nearest = nearest_neighbor(seed, G, vec, node_idx)
    nearest_cmus = node_cmus.get(nearest, [-1])

    if nearest_cmus[0] == -1:
        merge(nearest, cmu, node_cmus, cmu_nodes, 
              cmu_info, cmu_edges, G, vec, node_idx)

        merge_neighbor(nearest, cmu, G, vec, node_idx, node_cmus, cmu_nodes, 
                       cmu_info, cmu_edges)


def seed_cmu(seed, cmu_id, G, vec, node_idx, node_cmus, cmu_nodes, 
             cmu_info, cmu_edges):
    cur_cmu = cmu_id 

    nearest = nearest_neighbor(seed, G, vec, node_idx)
    nearest_cmus = node_cmus.get(nearest, [-1])

    if nearest_cmus[0] == -1:
        node_cmus[seed] = [cmu_id]
        cmu_nodes[cmu_id] = [seed]
        out_dist = 0
        for n in nx.neighbors(G, seed):
            out_dist += cal_dist(n, seed, vec, node_idx)
        cmu_info[cmu_id] = [0, out_dist]

        update_cmu_info(cmu_info, cmu_id, seed, G, 
                        node_cmus, vec, node_idx)

        cmu_edges[cmu_id] = [0, G.degree(seed)]

        update_cmu_edges(cmu_edges, cmu_id, seed, G, node_cmus)

        merge(nearest, cmu_id, node_cmus, cmu_nodes, 
              cmu_info, cmu_edges, G, vec, node_idx)

        merge_neighbor(nearest, cmu_id, G, vec, node_idx, node_cmus, 
                       cmu_nodes, cmu_info, cmu_edges)

        cmu_id = cmu_id + 1
    else:
        merge(seed, nearest_cmus[0], node_cmus, cmu_nodes, 
              cmu_info, cmu_edges, G, vec, node_idx)
        cur_cmu = nearest_cmus[0]

    return cur_cmu, cmu_id


def judge(mode, n, cmu, G, vec, node_idx, node_cmus, cmu_info, cmu_edges, alpha):
    if mode == 'dist':
        in_before, out_before = cmu_info.get(cmu) 
        in_after, out_after = cal_cmu_info(n, cmu, cmu_info, node_cmus, 
                                           G, vec, node_idx)
    elif mode == 'edges':
        in_before, out_before = cmu_edges.get(cmu) 
        in_after, out_after = cal_cmu_edges(cmu_edges, cmu, n, G, node_cmus)
    
    be = math.pow((in_before + out_before), alpha)
    af = math.pow((in_after + out_after), alpha)
    if be == 0 or af == 0:
        fitness_before = in_before
        fitness_after = fitness_before
    else:
        fitness_before = in_before/be
        fitness_after = in_after/af

    if fitness_after < fitness_before:
        if mode == 'dist':
            return True
        elif mode == 'edges':
            return False
    else:
        if mode == 'dist':
            return False
        elif mode == 'edges':
            return True


def expand_seed(cmu, G, vec, node_idx, cmu_nodes, node_cmus, 
                cmu_info, cmu_edges, alpha):
    nodes = cmu_nodes.get(cmu)
    for node in nodes:
        for n in nx.neighbors(G, node):
            n_cmus = node_cmus.get(n, [-1])

            if judge('dist', n, cmu, G, vec, node_idx, node_cmus, 
                     cmu_info, cmu_edges, alpha=alpha):
                if n_cmus[0] == -1:
                    merge(n, cmu, node_cmus, cmu_nodes, cmu_info, 
                          cmu_edges, G, vec, node_idx)
            
            if judge('edges', n, cmu, G, vec, node_idx, node_cmus, 
                     cmu_info, cmu_edges, alpha=alpha):
                if n_cmus[0] == -1:
                    merge(n, cmu, node_cmus, cmu_nodes, cmu_info, 
                          cmu_edges, G, vec, node_idx)


def expand(vector, G, node_idx, alpha=1.2):
    node_degree = [(n, d) for n, d in G.degree()]
    sorted_nodes = sorted(node_degree, key=lambda x: (x[1]), reverse=True)

    cmu_id = 0     
    node_cmus = {} 
    cmu_nodes = {} 
    cmu_info = {} 
    cmu_edges = {} # cmu_id:[in_edges, out_edges]

    for node, _ in sorted_nodes:
        if node_cmus.get(node) == None:
            cur_cmu, cmu_id = seed_cmu(node, cmu_id, G, vector, node_idx, 
                                       node_cmus, cmu_nodes, cmu_info, 
                                       cmu_edges)

            expand_seed(cur_cmu, G, vector, node_idx, cmu_nodes, node_cmus, 
                        cmu_info, cmu_edges, alpha=alpha)

    return cmu_nodes, node_cmus