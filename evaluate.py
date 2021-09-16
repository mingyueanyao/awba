# -*- coding: utf-8 -*-
import time
import argparse
import random
import warnings
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
np.set_printoptions(suppress = True)
 

def load_cmu_file(cmu_path):
    K = 0
    real_cmu = []
    node_number = 0
    with open(cmu_path, 'r') as f:
        for line in f:
            cmu = [int(node) for node in line.split()]
            node_number += len(cmu)
            real_cmu.append(cmu)
            K += 1

    real_label = np.zeros((node_number, ), dtype=int)
    for label, cmu in enumerate(real_cmu):
        for node in cmu:
            real_label[node] = label

    return real_label, K


def load_vec_file(vec_path):
    data = np.loadtxt(vec_path, dtype=bytes, delimiter=' ', skiprows=1)
    if data[0][-1] == b'':
        data = np.delete(data, -1, axis=1)
    data = data.astype(float)
    data = data[data[:, 0].argsort()]
    firstCol = data[:,0].astype(int)
    data = np.delete(data, 0, axis=1)

    return data


def cal_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def run(clf, data, real_label):
    s = clf.fit(data)
    predict_label = s.predict(data)[0]

    nmi = normalized_mutual_info_score(real_label, predict_label)
    ari = adjusted_rand_score(real_label, predict_label)
    ac = cal_acc(real_label, predict_label)

    return nmi, ari, ac
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--times', type=int, default=10)
    parser.add_argument('--cmu_path', type=str, default="./data/karate_community.txt")
    parser.add_argument('--vec_path', type=str, default="./output/karate_embeddings.txt")
    args = parser.parse_args()

    times = args.times
    cmu_path = args.cmu_path
    vec_path = args.vec_path
  
    start = time.time()

    real_label, K = load_cmu_file(cmu_path)
    data = load_vec_file(vec_path)
    clf = KMeans(n_clusters=K)

    nmi_list = []
    ari_list = []
    ac_list = []
    for t in range(times):
        nmi, ari, ac = run(clf, data, real_label)

        nmi_list.append(nmi)
        ari_list.append(ari)
        ac_list.append(ac)

    avg_nmi = sum(nmi_list) / times
    avg_ari = sum(ari_list) / times
    avg_ac = sum(ac_list) / times

    print("Kmeans on embeddings: {}\ncluster number: {}".format(vec_path, K))
    print('NMI (10 avg): {}\nARI (10 avg): {}\nAC (10 avg): {}'.format(avg_nmi, avg_ari, avg_ac))

    print("Running time: ", time.time() - start)