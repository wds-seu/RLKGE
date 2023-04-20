import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.pytorchtools import EarlyStopping, getDevice
from utils.metrics import precision_recall_f1, confusion
from utils.dataloader import  batch_by_num, batch_by_size
from model1.transE import TransEModule
import numpy as np
import matplotlib.pyplot as plt
import math


class ReinforcementModule(nn.Module):
    def __init__(self, config, relToCluIdx, dim):
        super().__init__()
        self.rel_num = len(relToCluIdx)
        self.relToCluIdx = relToCluIdx
        self.config = config
        self.dim = dim
        self.init_weight()

    def forward(self, input_embedding, rel):
        x = (self.rl_clu_embed(self.relToCluIdx[rel])+self.rl_rel_embed(rel))\
              *input_embedding
        x = torch.sum(x, dim=-1)
        return x

    def init_weight(self):
        self.rl_clu_embed = nn.Embedding(self.config['cluster'], self.dim)
        self.rl_rel_embed = nn.Embedding(self.rel_num, self.dim)

        # self.rl_clu_embed.weight.data.uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim))
        # self.rl_rel_embed.weight.data.uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim))

class ReinforcementModel(object):
    def __init__(self, config, relToCluIdx, kge_mdl:TransEModule, checkpoint_path):
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.kge_mdl = kge_mdl
        self.device = getDevice()
        self.relToCluIdx = relToCluIdx.to(self.device)
        self.mdl = ReinforcementModule(config, self.relToCluIdx, config['dim']).to(self.device)
        self.optimizer = Adam(self.mdl.parameters(), lr=5e-4)

    def train(self, train_data, avg_src_arr, avg_dst_arr, score_avg):
        src, rel, dst = train_data
        n_train = src.size(0)

        rand_idx = torch.randperm(n_train)
        src = src[rand_idx]
        rel = rel[rand_idx]
        dst = dst[rand_idx]

        loss = None
        # loss_all = 0
        step_loss = 0

        alpha = self.config['alpha']
        lambda1 = self.config['lambda1']
        lambda2 = self.config['lambda2']
        batch_num_rl = self.config['batch_num_rl']


        sampled_src = torch.tensor([], dtype=torch.int64, device=self.device)
        sampled_rel = torch.tensor([], dtype=torch.int64, device=self.device)
        sampled_dst = torch.tensor([], dtype=torch.int64, device=self.device)
        sample_num = 1e-30

        avg_src = avg_src_arr[rel[0].item()]
        avg_dst = avg_dst_arr[rel[0].item()]


        self.mdl.train()
        for s, r, t in batch_by_num(batch_num_rl, src, rel, dst):
            s, r, t = s.to(self.device), r.to(self.device), t.to(self.device)
            input_embedding = torch.cat([self.kge_mdl.rel_embed(r).data,
                                        self.kge_mdl.ent_embed(s).data,
                                        self.kge_mdl.ent_embed(t).data,
                                        torch.div(avg_src, sample_num+1).unsqueeze(0).expand(s.size(0), self.kge_mdl.dim),
                                        torch.div(avg_dst, sample_num+1).unsqueeze(0).expand(s.size(0), self.kge_mdl.dim)],
                                        dim=1)
            output= self.mdl.forward(input_embedding, r)

            # print(f'probs.data: {output.data}')

            probs = torch.sigmoid(output)
            actions = (probs > 0.5).view(-1, ).clone()
            action_int = actions.int()
            policy = (action_int - 0) * F.sigmoid(output) + (1-action_int)* (1- F.sigmoid(output))
            step_loss += torch.sum(torch.log(policy))

            # probs = torch.sigmoid(output)
            # actions = (probs > 0.5).view(-1, )

            # rewards = score_avg-(self.kge_mdl.forward(s, r, t))
            # loss = torch.sum(step_loss * rewards)
            # + lambda1 * torch.sum(self.mdl.rl_clu_embed(self.relToCluIdx[r[0]]) ** 2)\
            # + lambda2 * torch.sum(self.mdl.rl_rel_embed(r[0]) ** 2)
            #
            # self.optimizer.zero_grad()
            # loss.backward()
            # loss_all += loss.item()
            # self.optimizer.step()

            sampled_src = torch.cat([sampled_src, s[actions]], dim=0)
            sampled_rel = torch.cat([sampled_rel, r[actions]], dim=0)
            sampled_dst = torch.cat([sampled_dst, t[actions]], dim=0)
            sample_num += len(s[actions])

            avg_src += torch.sum(self.kge_mdl.ent_embed(s[actions]).data, dim=0)
            avg_dst += torch.sum(self.kge_mdl.ent_embed(t[actions]).data, dim=0)
        # print(f'train_data size: {n_train}, sampled size: {sampled_src.size(0)}')

        avg_src_arr[rel[0].item()] /= (sample_num+1)
        avg_dst_arr[rel[0].item()] /= (sample_num+1)

        # yield sampled_src, sampled_rel, sampled_dst

        if sample_num < 1:
            score = yield src, rel, dst
            reward = score / n_train
        else:
            score = yield sampled_src, sampled_rel, sampled_dst
            reward = score/sample_num + alpha * (sample_num/n_train)

        print(f'reward: {reward}')

        loss = reward * step_loss
        + lambda1 * torch.sum(self.mdl.rl_clu_embed(self.relToCluIdx[rel[0].to(self.device)]) ** 2)\
        + lambda2 * torch.sum(self.mdl.rl_rel_embed(rel[0].to(self.device)) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        yield loss.item()
        yield None

    def test(self, test_data, avg_src_arr, avg_dst_arr):
        src, rel, dst, label = test_data
        src, rel, dst = src.to(self.device), rel.to(self.device), dst.to(self.device)

        n_sample = src.size(0)
        preds = []

        avg_src = avg_src_arr[rel[0].item()]
        avg_dst = avg_dst_arr[rel[0].item()]


        for s, r, t in batch_by_size(256, src, rel, dst):
            input_embedding = torch.cat([self.kge_mdl.rel_embed(r).data,
                                         self.kge_mdl.ent_embed(s).data,
                                         self.kge_mdl.ent_embed(t).data,
                                         avg_src.unsqueeze(0).expand(s.size(0), self.kge_mdl.dim),
                                         avg_dst.unsqueeze(0).expand(s.size(0), self.kge_mdl.dim)],
                                         dim=1)
            output = self.mdl.forward(input_embedding, r)
            actions = (torch.sigmoid(output) > 0.5).long().view(-1,)
            preds.extend(actions.data.cpu())


        fn, tp, tn, fp = confusion(label, preds)
        print(f'tp: {tp} fp: {fp} tn: {tn} fn: {fn}', end='\t')
        precison = tp/(tp+fp)
        recall = tp/(tp+fn)
        sep= tn/(tn+fp)
        f1 = 2*precison*recall/(precison+recall)
        print(f'rl_model test result:  precison={precison}, recall={recall}, sep={sep} f1={f1}')




