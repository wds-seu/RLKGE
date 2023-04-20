import os
import json
import torch
from utils.read_data import *
from utils.config import getDatasetPathWithNoise
from utils.corrupter import BernCorrupter
from cluster import getCluster
from model1.transE import TransE
from rl import ReinforcementModel
from utils.pytorchtools import EarlyStopping, getDevice

def main(dataset=None, noiseRatio='N3'):
    # 载入数据，构建数据集
    train_path, test_path, valid_path, noise_path, reverseOfCol2_3 \
        = getDatasetPathWithNoise(dataset, noiseRatio)
    # 构建entity和relation的字典和list
    kb_index = indexing_ent_rel(train_path,
                                test_path,
                                valid_path,
                                noise_path,
                                reverseOfCol2_3=reverseOfCol2_3)
    print('build str2id index done!')
    n_ent, n_rel = graph_size(kb_index)
    # 构建数据集
    train_data = read_data(train_path, kb_index, reverseOfCol2_3)
    test_data = read_data(test_path, kb_index, reverseOfCol2_3)
    valid_data = read_data(valid_path, kb_index, reverseOfCol2_3)
    noise_data = read_data(noise_path, kb_index, reverseOfCol2_3)

    # 稀疏矩阵（这里构建稀疏矩阵应该不需要考虑noise_data）
    heads_sp, tails_sp = get_heads_tails_sparse(n_ent, train_data, valid_data, test_data)

    # 数据集tensor化
    train_tensor = [torch.LongTensor(vec) for vec in train_data]
    valid_tensor = [torch.LongTensor(vec) for vec in valid_data]
    test_tensor = [torch.LongTensor(vec) for vec in test_data]
    noise_tensor = [torch.LongTensor(vec) for vec in noise_data]

    # 构建负样本生成器应该不需要考虑noise_data
    corrupter = BernCorrupter(train_tensor, n_ent, n_rel)

    # 训练集添加标签和噪声数据
    train_tensor.append(torch.ones(train_tensor[0].size(0), dtype=torch.int64))
    noise_tensor.append(torch.zeros(noise_tensor[0].size(0), dtype=torch.int64))
    train_tensor_with_neg \
        = [torch.cat([pos, neg], dim=0) for pos, neg in zip(train_tensor, noise_tensor)]

    print('build dataset  done !')

    with open(''.join(['../config_', dataset, '.json']), 'r', encoding='utf-8') as f:
        configs = json.load(f)
    config_kge = configs['transE']
    config_rl = configs['rl_transE']
    kge_model = TransE(config_kge,
                   n_ent, n_rel,
                   checkpoint_path=os.path.join(os.path.dirname(train_path), 'transE'+noiseRatio+'.pth'))
    kge_tester = lambda: kge_model.test(valid_tensor, heads_sp, tails_sp)
    # print('kge train starting!')
    # kge_model.train(train_tensor_with_neg[:3], corrupter, kge_tester)
    # print('kge train done!')
    kge_model.load(os.path.join(os.path.dirname(train_path), 'transE'+noiseRatio+'.pth'))
    # kge_model.pca(train_tensor_with_neg)
    score_avg = kge_model.scoreDistributionScatter(train_tensor_with_neg)
    # kge_model.mdl.eval()
    # with torch.no_grad():
    #     kge_model.test(test_tensor, heads_sp, tails_sp)

    # relation 聚类分析
    print('kmeans processing')
    relToClu = getCluster(kge_model.mdl.rel_embed.weight.data.cpu().numpy(), n_clusters=config_rl['cluster'])
    relToClu = torch.LongTensor(relToClu)
    print('kmeans done')

    #按照关系分类划分训练集
    train_data_df, rel_set = getDataFrame(train_tensor_with_neg)

    # train_df, _ = getDataFrame(train_tensor)
    # noise_df, noi_set = getDataFrame(noise_tensor)

    rl_model = ReinforcementModel(config_rl,
                                  relToClu,
                                  kge_model.mdl,
                                  os.path.join(os.path.dirname(train_path), 'rl_for_transE'+noiseRatio+'.pth'))



    avg_src_arr = torch.tensor([[0.]], requires_grad=False).expand(len(relToClu), config_kge['dim']).to(getDevice())
    avg_dst_arr = torch.tensor([[0.]], requires_grad=False).expand(len(relToClu), config_kge['dim']).to(getDevice())

    #预训练rl模型
    print('rl pretrain starting')
    pretrain_episode = config_rl['pretrain_episode']
    for i in range(pretrain_episode):
        epoch_loss = 0
        sample_num = 0

        rel_random = np.random.choice(rel_set, size=len(rel_set), replace=False)
        for g in rel_random:
            data = train_data_df[train_data_df['rel'] == g]
            # print(f'sample for relation: {g}')
            #           rl-model sample action
            gen_step = rl_model.train([torch.from_numpy(data['src'].values),
                                      torch.from_numpy(data['rel'].values),
                                      torch.from_numpy(data['dst'].values)],
                                      avg_src_arr, avg_dst_arr,
                                      score_avg)
            sampled_train_tensor = next(gen_step)
            sampled_train_tensor = [tensor.cpu() for tensor in sampled_train_tensor]
            sample_num += len(sampled_train_tensor[0])
            #           kge training
            scores = kge_model.batch_score(sampled_train_tensor, config_rl['batch_kge'])
            #           rl-model参数更新
            loss = gen_step.send(scores)
            # loss = next(gen_step)
            epoch_loss += loss
        print(f'epoch: {i + 1} sample_num: {sample_num} loss: {epoch_loss/len(train_tensor_with_neg[0])}')

    rl_model.mdl.eval()
    with torch.no_grad():
        score_avg = kge_model.scoreDistributionScatter(train_tensor_with_neg)
        rl_model.test(train_tensor_with_neg, avg_src_arr, avg_dst_arr)
    print('rl pretrain done')

    #联合训练
    print('rl train starting')
    episode = config_rl['episode']
    for i in range(episode):
        epoch_loss = 0
        sample_num = 0

        rel_random = np.random.choice(rel_set, size=len(rel_set), replace=False)
        for g in rel_random:
            data = train_data_df[train_data_df['rel'] == g]
            # print(f'sample for relation: {key}')
#           rl-model sample action
            gen_step = rl_model.train([torch.from_numpy(data['src'].values),
                  torch.from_numpy(data['rel'].values),
                  torch.from_numpy(data['dst'].values)],
                  avg_src_arr, avg_dst_arr,
                  score_avg)
            sampled_train_tensor = next(gen_step)
            sampled_train_tensor = [tensor.cpu() for tensor in sampled_train_tensor]
            sample_num += len(sampled_train_tensor[0])
#           kge training
            scores = kge_model.one_epoch_train(sampled_train_tensor, corrupter, config_rl['batch_kge'])
#           rl-model参数更新
            loss = gen_step.send(scores)
            # loss = next(gen_step)
            epoch_loss += loss
        print(f'epoch: {i+1} sample_num: {sample_num} loss: {epoch_loss/len(train_tensor_with_neg[0])}')

#       validation：test_link(kge)
        rl_model.mdl.eval()
        kge_model.mdl.eval()
        with torch.no_grad():
            kge_tester()
            score_avg = kge_model.scoreDistributionScatter(train_tensor_with_neg)
            rl_model.test(train_tensor_with_neg,avg_src_arr, avg_dst_arr)

#   保存模型(transE_advancedByRl.pth; rl_for_transE.pth)  加载模型
    print('rl train done')

    rl_model.mdl.eval()
    kge_model.mdl.eval()
    with torch.no_grad():
        kge_model.test(test_tensor, heads_sp, tails_sp)
        kge_model.scoreDistributionScatter(train_tensor_with_neg)
        rl_model.test(train_tensor_with_neg,avg_src_arr, avg_dst_arr)
#   testing：error detection(rl-model)
#   testing：test_link(kge)

if __name__ == "__main__":
    main(dataset="fb15k")