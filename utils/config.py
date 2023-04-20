
def getDatasetPath(dataset):
    if dataset == 'fb15k':
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/fb15k/FB15k/'
        return prefix+'train.txt', prefix+'test.txt', prefix+'valid.txt', False
    elif dataset == 'fb15k237':
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/fb15k237/'
        return prefix + 'train.txt', prefix + 'test.txt', prefix + 'valid.txt', True
    elif dataset == 'wn18rr':
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/wn18rr/'
        return prefix + 'train.txt', prefix + 'test.txt', prefix + 'valid.txt', True
    else:
        raise NotImplementedError


def getDatasetPathWithNoise(dataset, noiseRatio):
    train_path, test_path, valid_path, reverseOfCol2_3 = getDatasetPath(dataset)
    if dataset == 'fb15k':
        print('use dataset fb15k')
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/fb15k/'
        noise_path =  prefix+noiseRatio+'/neg_train.txt'
    elif dataset == 'fb15k237':
        print('use dataset fb15k237')
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/fb15k237/'
        noise_path =  prefix+noiseRatio+'/neg_train.txt'
    elif dataset == 'wn18rr':
        print('use dataset wn18rr')
        prefix = '/home/wds/zhangjiong/Integration-model/dataset/wn18rr/'
        noise_path =  prefix+noiseRatio+'/neg_train.txt'
    else:
        raise NotImplementedError
    return train_path, test_path, valid_path,noise_path, reverseOfCol2_3