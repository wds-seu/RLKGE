import os

dir = '/home/wds/zhangjiong/Integration-model/dataset/fb15k237'
entity2text_file = '/home/wds/zhangjiong/Integration-model/dataset/fb15k237/entity2text.txt'
relation2text_file = '/home/wds/zhangjiong/Integration-model/dataset/fb15k237/relation2text.txt'
files = []
exclude_file = ['entity2text.txt', 'relation2text.txt']

def gci(filepath):
    fs = os.listdir(filepath)
    for fi in fs:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            gci(fi_d)
        elif os.path.splitext(fi_d)[1] == '.txt' and fi not in exclude_file:
                files.append(fi_d)
                # print(fi_d)


def index_str_text(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            str, text = line.strip().split('\t')
            dict[str] = text
    return dict

def make_claims(files,entity_dict, relation_dict):
    for file in files:
        filepath, filename = os.path.split(file)
        name, suffix = os.path.splitext(filename)
        claim_file = os.path.join(filepath, name+'_claim'+ suffix)
        print(claim_file)
        claim_file = open(claim_file, 'w')
        with open(file, 'r') as f:
            for line in f:
                s, r, t = line.strip().split('\t')
                claim_file.write('\t'.join([entity_dict[s], relation_dict[r], entity_dict[t]]))
                claim_file.write('\n')

gci(dir)
entity_dict = index_str_text(entity2text_file)
relation_dict = index_str_text(relation2text_file)
make_claims(files, entity_dict, relation_dict)