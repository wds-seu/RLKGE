from itertools import count
from collections import namedtuple, defaultdict
import torch
import pandas as pd
import numpy as np

KBIndex = namedtuple('KBIndex', ['ent_list', 'rel_list', 'ent_id', 'rel_id', 'id_ent', 'id_rel'])

# str -> id
# reverseOfCol2_3: false, 数据集顺序：头实体 尾实体 关系; true, 数据集顺序：头实体 关系  尾实体
def indexing_ent_rel(*filenames, reverseOfCol2_3):
    ent_set = set()
    rel_set = set()
    for file in filenames:
        with open(file, 'r') as f:
            for line in f:
                if reverseOfCol2_3:
                    s, r, t =line.strip().split('\t')
                else:
                    s, t, r = line.strip().split('\t')
                ent_set.add(s)
                ent_set.add(t)
                rel_set.add(r)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    id_ent = dict(zip(count(), ent_list))
    id_rel = dict(zip(count(), rel_list))
    return KBIndex(ent_list, rel_list, ent_id, rel_id, id_ent, id_rel)

def graph_size(kb_index :KBIndex):
    return len(kb_index.ent_id), len(kb_index.rel_id)


def read_data(filename:str, kb_index:KBIndex, reverseOfCol2_3):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            if reverseOfCol2_3:
                s, r, t = line.strip().split('\t')
            else:
                s, t, r = line.strip().split('\t')
            src.append(kb_index.ent_id[s])
            dst.append(kb_index.ent_id[t])
            rel.append(kb_index.rel_id[r])
    return src, rel, dst

def get_heads_tails_sparse(n_ent, train_data, valid_data = None, test_data = None):
    train_src, train_rel, train_dst = train_data
    if valid_data:
        valid_src, valid_rel, valid_dst = valid_data
    else:
        valid_src = valid_rel = valid_dst = []
    if test_data:
        test_src, test_rel, test_dst = test_data
    else:
        test_src = test_rel = test_dst = []
    all_src = train_src + valid_src + test_src
    all_rel = train_rel + valid_rel + test_rel
    all_dst = train_dst + valid_dst + test_dst
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for s, r, t in zip(all_src, all_rel, all_dst):
        tails[(s, r)].add(t)
        heads[(t, r)].add(s)
    heads_sp = {}
    tails_sp = {}
    for k in tails.keys():
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                               torch.ones(len(tails[k])), torch.Size([n_ent]))
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                               torch.ones(len(heads[k])), torch.Size([n_ent]))
    return heads_sp, tails_sp

def getDataFrame(train_data):
    src, rel, dst, label = train_data
    data = pd.DataFrame({"src": src, "rel": rel, "dst":dst, "label":label})
    rel_set = data['rel'].unique()
    # for g in group:
    #     # key = g[0]
    #     val = g[1]
    #     print(torch.from_numpy(val['src'].values))
    #     print(torch.from_numpy(val['rel'].values))
    #     print(torch.from_numpy(val['dst'].values))
    #     break
    return data, rel_set


