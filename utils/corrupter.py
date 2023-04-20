import numpy.random
import torch
from collections import defaultdict


def get_bern_prob(data, n_rel):
    src, rel, dst = data
    edges = defaultdict(lambda: defaultdict(lambda: set()))
    rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
    for s, r, t in zip(src, rel, dst):
        edges[r.item()][s.item()].add(t.item())
        rev_edges[r.item()][t.item()].add(s.item())

    bern_prob = torch.zeros(n_rel)
    for r in edges.keys():
        tph = sum(len(tail_set) for tail_set in edges[r].values()) / len(edges[r])
        hph = sum(len(head_set) for head_set in rev_edges[r].values()) / len(rev_edges[r])
        bern_prob[r] = tph / (tph + hph)
    return bern_prob


class BernCorrupter():
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = get_bern_prob(data, n_rel)
        self.n_ent = n_ent
        self.data = data
    def corrupt(self, src, rel, dst):
        prob = self.bern_prob[rel]
        selection = torch.bernoulli(prob).type(torch.int64)
        ent_random = numpy.random.choice(self.n_ent, len(src))
        src_out = (1-selection) * src + selection * ent_random
        dst_out = selection * dst + (1-selection) * ent_random
        return src_out, dst_out



if __name__  == "__main__":
    pass
