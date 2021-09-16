
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product, permutations
from scipy.io import loadmat
from scipy.sparse import issparse

logger = logging.getLogger("deepwalk")
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"




class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        return len(self)

    def number_of_edges(self):
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        return self.order()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None, 
                    node_cmu_nodes=None, att_mtx=None, node_idx=None, cmu_walk_p=1.0):
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1] 
            next_nodes = G[cur]
            
            next_nodes_p = [1.0]*len(next_nodes)
            if att_mtx is not None:
                cur_id = node_idx[str(cur)]
                next_nodes_p = [float(att_mtx[cur_id][node_idx[str(nid)]]) for nid in next_nodes]

            if node_cmu_nodes is not None:
                for i, n in enumerate(next_nodes):
                    if str(n) in node_cmu_nodes:
                        next_nodes_p[i] *= cmu_walk_p
                    else:
                        next_nodes_p[i] *= (1-cmu_walk_p)
                
            if len(next_nodes) > 0:
                if rand.random() >= alpha:
                    if sum(next_nodes_p) != 0:
                        next_node = rand.choices(next_nodes, weights=next_nodes_p, k=1)
                        path.append(next_node[0])
                    else:
                        break
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0), chunk=0, nodes=None,
                               cmu_nodes=None, node_cmu=None, att_mtx=None, 
                               node_idx=None, cmu_walk_p=1.0):

    nodes = nodes[chunk * num_paths: (chunk + 1) * num_paths]
    for node in nodes:
        node_cmu_nodes = None
        if cmu_nodes != None:
            cmu_id = node_cmu[str(node)]
            node_cmu_nodes = cmu_nodes[cmu_id]  
        yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node, 
                            node_cmu_nodes=node_cmu_nodes, att_mtx=att_mtx, 
                            node_idx=node_idx, cmu_walk_p=cmu_walk_p)


def load_edgelist_from_csr_matrix(adjList, firstcol, undirected=True):
    G = Graph()
    for idx, ads in enumerate(adjList.tolil().rows):
        x = int(firstcol[idx])
        for idy in ads:
            y = int(firstcol[idy])
            if y not in G[x]:
                G[x].append(y)
            if undirected and (x not in G[y]):
                G[y].append(x)
        
    G.make_consistent()
    return G