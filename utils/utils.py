import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_distance

def label_onehot(inputs, num_segments):

    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((batch_size, num_segments, im_h, im_w)).cuda()

    inputs_temp = inputs.unsqueeze(1).clone().type(torch.int64)
    outputs.scatter_(1, inputs_temp, 1.0).type(torch.int64)

    return outputs


def one_hot2dist(one_hot):

    one_hot = one_hot.cpu().numpy() 
    B,K,H,W = one_hot.shape

    res = np.zeros_like(one_hot)
    for b in range(B):
        for k in range(K):
            posmask = one_hot[b][k].astype(np.bool)

            if posmask.any():
                negmask = ~posmask
                res[b][k] = eucl_distance(negmask) * negmask \
                    - (eucl_distance(posmask) - 1) * posmask
            # The idea is to leave blank the negative classes
            # since this is one-hot encoded, another class will supervise that pixel
    res = torch.from_numpy(res).cuda()

    return res


def cal_category_confidence(preds_student_sup, gt, num_classes):
    category_confidence = torch.zeros(num_classes).type(torch.float32)
    preds_student_sup = F.softmax(preds_student_sup, dim=1)
    for ind in range(num_classes):
        cat_mask_sup_gt = (gt == ind)
        if torch.sum(cat_mask_sup_gt) == 0:
            value = 0
        else:
            conf_map_sup = preds_student_sup[:, ind, :, :]
            value = torch.sum(conf_map_sup*cat_mask_sup_gt)/(torch.sum(cat_mask_sup_gt)+1e-12)
        category_confidence[ind] = value

    return category_confidence



