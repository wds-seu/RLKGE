import torch
from sklearn.metrics import precision_score, recall_score, confusion_matrix
def mrr_mr_hitk(scores, target, k=10):
    _, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target.cuda()
    # print(find_target)
    target_rank = torch.nonzero(find_target)[0, 0] + 1
    target_rank = target_rank.item()
    return 1 / target_rank, target_rank, int(target_rank <= k)

def precision_recall_f1(y_true, y_pred):
    precison = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2*precison*recall/(precison+recall)
    return precison, recall, f1

def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return fn, tp, tn, fp