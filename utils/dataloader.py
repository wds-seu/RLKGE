
def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    if (n_batch > n_sample):
        if len(lists) > 1:
            yield lists
        else:
            yield lists[0]
    else:
        for i in range(n_batch):
            head = int(n_sample * i / n_batch)
            tail = int(n_sample * (i + 1) / n_batch)
            ret = [ls[head:tail] for ls in lists]
            if len(ret) > 1:
                yield ret
            else:
                yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

if __name__ == "__main__":
    pass