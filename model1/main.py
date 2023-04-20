import json
import os
from utils.read_data import *
from utils.config import getDatasetPath
from utils.corrupter import BernCorrupter
from transE import TransE

def main(dataset=None):
    # 载入数据，构建数据集
    train_path, test_path, valid_path, reverseOfCol2_3 = getDatasetPath(dataset)
    # 构建entity和relation的字典和list
    kb_index = indexing_ent_rel(train_path,
                                test_path,
                                valid_path,
                                reverseOfCol2_3=reverseOfCol2_3)
    print('build str2id index done!')
    n_ent, n_rel = graph_size(kb_index)
    train_data = read_data(train_path, kb_index, reverseOfCol2_3)
    test_data = read_data(test_path, kb_index, reverseOfCol2_3)
    valid_data = read_data(valid_path, kb_index, reverseOfCol2_3)

    heads_sp, tails_sp = get_heads_tails_sparse(n_ent, train_data, valid_data, test_data)


    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data = [torch.LongTensor(vec) for vec in test_data]


    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    print('build dataset  done !')

    with open(''.join(['../config_', dataset, '.json']), 'r', encoding='utf-8') as f:
        configs = json.load(f)
    print(configs)
    model = TransE(configs['transE'],
                   n_ent, n_rel,
                   checkpoint_path=os.path.join(os.path.dirname(train_path), 'transE.pth'))

    tester = lambda: model.test(valid_data, heads_sp, tails_sp)
    print('train beginning!')
    model.train(train_data, corrupter, tester)
    print('train done!')
    model.load(os.path.join(os.path.dirname(train_path), 'transE.pth'))
    model.mdl.eval()
    with torch.no_grad():
        model.test(test_data, heads_sp, tails_sp)




if __name__ == "__main__":
    main(dataset="fb15k")
    # 载入数据，构建数据集
    # train_path, test_path, valid_path, reverseOfCol2_3 = getDatasetPath("fb15k237")
    # # 构建entity和relation的字典和list
    # kb_index = indexing_ent_rel(train_path,
    #                             test_path,
    #                             valid_path,
    #                             reverseOfCol2_3=reverseOfCol2_3)
    # n_ent, n_rel = graph_size(kb_index)
    # train_data = read_data(train_path, kb_index, reverseOfCol2_3)
    # test_data = read_data(test_path, kb_index, reverseOfCol2_3)
    # valid_data = read_data(valid_path, kb_index, reverseOfCol2_3)
    #
    # heads_sp, tails_sp = get_heads_tails_sparse(n_ent, train_data, valid_data, test_data)
    #
    # train_data = [torch.LongTensor(vec) for vec in train_data]
    #
    # corrupter = BernCorrupter(train_data, n_ent, n_rel)
    # src_corrupted, dst_corrupted = corrupter.corrupt()
    # src, rel, dst = train_data
    # dataGen = batch_by_num(100, src, rel, dst, src_corrupted, dst_corrupted)
    # src_bat, rel_bat, dst_bat, src_corrupted_bat, dst_corrupted_bat = [i[:30] for i in next(dataGen)]
    # for a,b,c,d,e in zip(src_bat, rel_bat, dst_bat, src_corrupted_bat, dst_corrupted_bat):
    #     print(a,b,c,d,e,sep='\t')