import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.pytorchtools import EarlyStopping, getDevice
from utils.metrics import mrr_mr_hitk
from utils.dataloader import  batch_by_num, batch_by_size
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class TransEModule(nn.Module):
    def __init__(self, dim, p , n_ent, n_rel):
        super().__init__()
        self.dim = dim
        self.p = p
        self.rel_embed = nn.Embedding(n_rel, self.dim)
        self.ent_embed = nn.Embedding(n_ent, self.dim)
        self.init_weight()


    def forward(self, src, rel, dst):
        ret =  torch.norm(self.ent_embed(dst) -self.ent_embed(src)- self.rel_embed(rel)  + 1e-30,
                            p=self.p,
                            dim=-1)
        return ret

    def distance(self, src, rel, dst):
        return self.ent_embed(dst) -self.ent_embed(src)- self.rel_embed(rel)

    def init_weight(self):
        self.ent_embed.weight.data.uniform_(-6/math.sqrt(self.dim), 6/math.sqrt(self.dim))
        self.rel_embed.weight.data.uniform_(-6/math.sqrt(self.dim), 6/math.sqrt(self.dim))
        self.rel_embed.weight.data = F.normalize(self.rel_embed.weight.data, p=self.p, dim=1)

    def constraint(self):
        # self.ent_embed.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        # self.rel_embed.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.ent_embed.weight.data.renorm_(2, 0, 1)
        self.rel_embed.weight.data.renorm_(2, 0, 1)

class TransE(object):
    def __init__(self, config, n_ent, n_rel, checkpoint_path):
        super().__init__()
        self.device = getDevice()
        self.mdl = TransEModule(config['dim'], config['p'], n_ent, n_rel).to(self.device)
        self.margin = int(config['margin'])
        self.n_epoch = int(config['n_epoch'])
        self.epoch_per_test = int(config['epoch_per_test'])
        # self.heads_sp, self.tails_sp = heads_sp, tails_sp
        self.n_ent, self.n_rel = n_ent, n_rel
        self.checkpoint_path = checkpoint_path
        self.early_stopping = EarlyStopping(patience=5, verbose=True, path=self.checkpoint_path)

    def train(self, train_data, corrupter, tester):
        src, rel, dst= train_data
        optimizer = Adam(self.mdl.parameters(), lr = 1e-3)
        n_train = len(src)
        for epoch in range(self.n_epoch):
            self.mdl.train()
            epoch_loss = 0
            rand_idx = torch.randperm(n_train)
            src = src[rand_idx]
            rel = rel[rand_idx]
            dst = dst[rand_idx]
            src_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
            for s0, r, t0, s1, t1 in batch_by_num(100, src, rel, dst, src_corrupted, dst_corrupted, n_sample=n_train):
                s0, r, t0, s1, t1 = s0.to(self.device), r.to(self.device), t0.to(self.device),\
                    s1.to(self.device), t1.to(self.device)
                self.mdl.zero_grad()
                loss = torch.sum(F.relu(self.mdl.forward(s0, r, t0) + self.margin - self.mdl.forward(s1, r, t1)))
                loss.backward()
                optimizer.step()
                self.mdl.constraint()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.n_epoch}, Loss={epoch_loss / n_train}")
            if (epoch + 1) % self.epoch_per_test == 0:
                self.mdl.eval()
                with torch.no_grad():
                    test_perf = tester()
                    self.early_stopping(test_perf, self.mdl)
                    if self.early_stopping.early_stop:
                        print('train finish,  early stop')
                        break


    def one_epoch_train(self, train_data, corrupter, batch_size):
        src, rel, dst= train_data
        optimizer = Adam(self.mdl.parameters(), lr = 1e-3)
        n_train = len(src)
        self.mdl.train()
        score_all = 1e-30
        rand_idx = torch.randperm(n_train)
        src = src[rand_idx]
        rel = rel[rand_idx]
        dst = dst[rand_idx]
        src_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
        for s0, r, t0, s1, t1 in batch_by_size(batch_size, src, rel, dst, src_corrupted, dst_corrupted, n_sample=n_train):
            s0, r, t0, s1, t1 = s0.to(self.device), r.to(self.device), t0.to(self.device), \
                s1.to(self.device), t1.to(self.device)
            score = self.mdl.forward(s0, r, t0)
            fake_score = self.mdl.forward(s1, r, t1)
            score_all += torch.sum(score).item()
            loss = torch.sum(F.relu(score + self.margin - fake_score))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.mdl.constraint()
        return -score_all

    def batch_score(self, train_data, batch_size):
        src, rel, dst= train_data
        n_train = len(src)
        self.mdl.eval()
        score_all = 1e-30
        for s0, r, t0 in batch_by_size(batch_size, src, rel, dst, n_sample=n_train):
            s0, r, t0= s0.to(self.device), r.to(self.device), t0.to(self.device)
            score = self.mdl.forward(s0, r, t0)
            score_all += torch.sum(score).item()
        return -score_all

    def scoreDistributionScatter(self, test_data):
        src, rel, dst, label = test_data
        src, rel, dst = src.to(self.device), rel.to(self.device), dst.to(self.device)
        preds = []
        for s, r, t in batch_by_size(256, src, rel, dst):
            output = self.mdl.forward(s,r,t)
            preds.extend(output.data.cpu())
        map_color = {0: 'r', 1: 'g'}
        color = list(map(lambda  x: map_color[x.item()], label))
        plt.scatter(np.arange(1, len(src)+1), preds, s = 1, c=color)
        plt.show()

        positive_avg = sum((map(lambda  x, y: (y if x == 1 else 0), label, preds))) / torch.sum(label)
        negative_avg = sum((map(lambda  x, y: (y if x == 0 else 0), label, preds))) / (len(label) - torch.sum(label))
        print(f'positive_avg: {positive_avg} nagetive_avg: {negative_avg}')
        return np.mean(preds)

    def pca(self, test_data):
        src, rel, dst, label = test_data
        src, rel, dst = src.to(self.device), rel.to(self.device), dst.to(self.device)
        outputs = []
        pca = PCA(n_components=2)
        for s, r, t in batch_by_size(256, src, rel, dst):
            output = self.mdl.distance(s,r,t)
            outputs.extend(output.data.cpu().numpy())

        pca_res = pca.fit_transform(outputs)
        map_color = {0: 'r', 1: 'g'}
        color = list(map(lambda x: map_color[x.item()], label))
        plt.scatter(pca_res[:,0], pca_res[:,1], s=1, c=color)
        plt.show()


    def load(self, checkpoint_path):
        print('model state_dict:', [key for key in self.mdl.state_dict().keys()])
        self.mdl.load_state_dict(torch.load(checkpoint_path))
    def test(self, test_data, heads_sp, tails_sp, filt = True):
        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        src, rel, dst = test_data
        for batch_s, batch_r, batch_t in batch_by_size(128, src, rel, dst):
            batch_size = batch_s.size(0)
            # self.mdl.forward(batch_s.to(self.device), batch_r.to(self.device), batch_t.to(self.device))
            batch_s, batch_r, batch_t = batch_s.to(self.device), batch_r.to(self.device), batch_t.to(self.device)

            src_expand = batch_s.unsqueeze(1).expand(batch_size, self.n_ent)
            rel_expand = batch_r.unsqueeze(1).expand(batch_size, self.n_ent)
            dst_expand = batch_t.unsqueeze(1).expand(batch_size, self.n_ent)
            all_var = torch.arange(0, self.n_ent).unsqueeze(0).expand(batch_size, self.n_ent).\
                type(torch.LongTensor).to(self.device)
            dst_scores = self.mdl.forward(src_expand, rel_expand, all_var).detach()
            src_scores = self.mdl.forward(all_var, rel_expand, dst_expand).detach()
            for s, r, t, dst_score, src_score in zip(batch_s, batch_r, batch_t, dst_scores, src_scores):
                if filt:
                    if tails_sp[(s.item(), r.item())]._nnz() > 1:
                        tmp = dst_score[t].item()
                        dst_score += tails_sp[(s.item(), r.item())].to(self.device) * 1e30
                        dst_score[t] = tmp
                    if heads_sp[(t.item(), r.item())]._nnz() > 1:
                        tmp = src_score[s].item()
                        src_score += heads_sp[(t.item(), r.item())].to(self.device) * 1e30
                        src_score[s] = tmp
                mrr, mr, hit10 = mrr_mr_hitk(dst_score, t)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                mrr, mr, hit10 = mrr_mr_hitk(src_score, s)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                count += 2
        print(f'transE test result:  Test_MRR={mrr_tot / count}, Test_MR={mr_tot / count}, Test_H@10={hit10_tot / count}')
        return mrr_tot / count





