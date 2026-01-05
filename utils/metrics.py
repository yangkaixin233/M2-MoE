import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import networkx as nx


def cal_auc(scores, trues):
    auc = roc_auc_score(trues, scores)
    return auc


def cal_f1_macro(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    macro_f1 = f1_score(trues, preds, average='macro')
    return macro_f1


def cal_recall(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    recall = recall_score(trues, preds, average='macro')
    return recall


def cal_shortest_dis(edge_index):
    dis_shortest = {}
    edge_index_ = edge_index.cpu().numpy().astype(int).tolist()
    G = nx.Graph()
    for i in range(len(edge_index_[0])):
        G.add_edge(edge_index_[0][i], edge_index_[1][i])
    d = dict(nx.shortest_path_length(G))
    for i in range(len(edge_index_[0])):
        dis = d[edge_index_[0][i]][edge_index_[1][i]]
        if dis == 0:
            dis = np.inf
        dis_shortest[(edge_index_[0][i], edge_index_[1][i])] = dis
        dis_shortest[(edge_index_[1][i], edge_index_[0][i])] = dis

    return dis_shortest
