import torch
import numpy as np


def adv_check_and_update(X_cur, logits, y, not_correct, X_adv):
    adv_pred = logits.max(1)[1]
    nc = (adv_pred != y.data)
    not_correct += nc.long()
    X_adv[nc] = X_cur[nc].detach()
    return X_adv, not_correct


def one_hot_tensor(y_batch_tensor, num_classes):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor
