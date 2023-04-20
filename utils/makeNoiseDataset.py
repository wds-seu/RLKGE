import random
import os
from utils.config import getDatasetPath
from utils.read_data import indexing_ent_rel, graph_size, read_data
from collections import  defaultdict
import numpy as np

dataset = 'fb15k237'
ratio = [0.1, 0.2, 0.4]
filename_list = ['N1', 'N2', 'N3']


train_path, test_path, valid_path, reverseOfCol2_3 = getDatasetPath(dataset)
# 构建entity和relation的字典和list
kb_index = indexing_ent_rel(train_path,
                            test_path,
                            valid_path,
                            reverseOfCol2_3=reverseOfCol2_3)
print('build str2id index done!')
train_data = read_data(train_path, kb_index, reverseOfCol2_3)
test_data = read_data(test_path, kb_index, reverseOfCol2_3)
valid_data = read_data(valid_path, kb_index, reverseOfCol2_3)

id_to_ent = kb_index.id_ent
id_to_rel = kb_index.id_rel

train_src, train_rel, train_dst = train_data
valid_src, valid_rel, valid_dst = valid_data
test_src, test_rel, test_dst = test_data
all_src = train_src + valid_src + test_src
all_rel = train_rel + valid_rel + test_rel
all_dst = train_dst + valid_dst + test_dst

rel_src_dict = {}
rel_dst_dict = {}

for s, r, t in zip(all_src, all_rel, all_dst):
    if r in rel_src_dict:
        rel_src_dict[r].add(s)
    else:
        rel_src_dict[r] = set()
    if r in rel_dst_dict:
        rel_dst_dict[r].add(t)
    else:
        rel_dst_dict[r] = set()

for key in rel_src_dict:
    rel_src_dict[key] = list(rel_src_dict[key])
    rel_dst_dict[key] = list(rel_dst_dict[key])

for  rt, fn in zip(ratio, filename_list):
    print(f'make noise dataset: {fn}')
    noise_idx = np.random.choice(range(len(all_rel)), size=int(rt * len(all_rel)), replace=True)
    n_src, n_rel, n_dst = [],[],[]
    for i, r in enumerate(noise_idx):
        if i % 2 == 0:
            if len(rel_dst_dict[all_rel[r]]) > 1:
                rt_dst = random.choice(rel_dst_dict[all_rel[r]])
                while(rt_dst == all_dst[r]):
                    rt_dst = random.choice(rel_dst_dict[all_rel[r]])
                n_src.append(all_src[r])
                n_rel.append(all_rel[r])
                n_dst.append(rt_dst)
        else:
            if len(rel_src_dict[all_rel[r]]) > 1:
                rt_src = random.choice(rel_src_dict[all_rel[r]])
                while (rt_src == all_src[r]):
                    rt_src = random.choice(rel_src_dict[all_rel[r]])
                n_src.append(rt_src)
                n_rel.append(all_rel[r])
                n_dst.append(all_dst[r])

    print(f'len of noise_data: {len(n_src)}')

    test_num = int(len(n_src) * 0.1)
    valid_num = int(len(n_src) * 0.2)
    n_src_train, n_src_test,n_src_valid = n_src[valid_num:], n_src[:test_num], n_src[test_num:valid_num]
    n_rel_train, n_rel_test,n_rel_valid = n_rel[valid_num:], n_rel[:test_num], n_rel[test_num:valid_num]
    n_dst_train, n_dst_test,n_dst_valid = n_dst[valid_num:], n_dst[:test_num], n_dst[test_num:valid_num]

    with open(os.path.dirname(train_path) + '/'+ fn + '/neg_train.txt', 'w') as f:
        for s, r, t in zip(n_src_train, n_rel_train, n_dst_train):
            f.write('\t'.join([id_to_ent[s], id_to_rel[r], id_to_ent[t]]))
            f.write('\n')
        f.close()
    with open(os.path.dirname(train_path) + '/'+ fn + '/neg_test.txt', 'w') as f:
        for s, r, t in zip(n_src_test, n_rel_test, n_dst_test):
            f.write('\t'.join([id_to_ent[s], id_to_rel[r], id_to_ent[t]]))
            f.write('\n')
        f.close()
    with open(os.path.dirname(train_path) + '/' + fn + '/neg_valid.txt', 'w') as f:
        for s, r, t in zip(n_src_valid, n_rel_valid, n_dst_valid):
            f.write('\t'.join([id_to_ent[s], id_to_rel[r], id_to_ent[t]]))
            f.write('\n')
        f.close()