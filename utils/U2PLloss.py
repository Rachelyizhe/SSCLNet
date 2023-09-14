import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
import copy
from torch.nn import functional as F


from .utils import label_onehot


def compute_rce_loss(predict, target):
    from einops import rearrange

    predict = F.softmax(predict, dim=1)

    with torch.no_grad():
        _, num_cls, h, w = predict.shape
        temp_tar = target.clone()
        temp_tar[target == 255] = 0

        label = (
            F.one_hot(temp_tar.clone().detach(), num_cls).float().cuda()
        )  # (batch, h, w, num_cls)
        label = rearrange(label, "b h w c -> b c h w")
        label = torch.clamp(label, min=1e-4, max=1.0)

    rce = -torch.sum(predict * torch.log(label), dim=1) * (target != 255).bool()
    return rce.sum() / (target != 255).sum()




def compute_qualified_pseudo_label(target, percent, pred_teacher):

    with torch.no_grad():
        # drop pixels with high entropy
        num = torch.sum(target != 0)
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        if torch.sum(target == 1) > 0:
            thresh1 = np.percentile(
                entropy[target == 1].detach().cpu().numpy().flatten(), percent[0]
            )
            thresh_mask = entropy.ge(thresh1).bool() * (target == 1).bool()
            target[thresh_mask] = 0

        if torch.sum(target == 2) > 0:
            thresh2 = np.percentile(
                entropy[target == 2].detach().cpu().numpy().flatten(), percent[1]
            )
            thresh_mask = entropy.ge(thresh2).bool() * (target == 2).bool()
            target[thresh_mask] = 0

        if torch.sum(target == 3) > 0:
            thresh3 = np.percentile(
                entropy[target == 3].detach().cpu().numpy().flatten(), percent[2]
            )
            thresh_mask = entropy.ge(thresh3).bool() * (target == 3).bool()
            target[thresh_mask] = 0

        if torch.sum(target == 4) > 0:
            thresh4 = np.percentile(
                entropy[target == 4].detach().cpu().numpy().flatten(), percent[3]
            )
            thresh_mask = entropy.ge(thresh4).bool() * (target == 4).bool()
            target[thresh_mask] = 0

        if torch.sum(target == 5) > 0:
            thresh5 = np.percentile(
                entropy[target == 5].detach().cpu().numpy().flatten(), percent[4]
            )
            thresh_mask = entropy.ge(thresh5).bool() * (target == 5).bool()
            target[thresh_mask] = 0

        if torch.sum(target == 6) > 0:
            thresh6 = np.percentile(
                entropy[target == 6].detach().cpu().numpy().flatten(), percent[5]
            )
            thresh_mask = entropy.ge(thresh6).bool() * (target == 6).bool()
            target[thresh_mask] = 0

        weight = num / torch.sum(target != 0)

    return target, weight


def compute_unsupervised_loss_U2PL(predict, target, weight):

    print()
    loss = weight * F.cross_entropy(predict, target-1, weight=torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda(), ignore_index=-1)  # [10, 321, 321]

    return loss


# def compute_unsupervised_loss_U2PL(predict, target, percent, pred_teacher):

#     batch_size, num_class, h, w = predict.shape

#     with torch.no_grad():
#         # drop pixels with high entropy
#         num = torch.sum(target != 0)
#         prob = torch.softmax(pred_teacher, dim=1)
#         entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

#         thresh = np.percentile(
#             entropy[target != 0].detach().cpu().numpy().flatten(), percent
#         )
#         thresh_mask = entropy.ge(thresh).bool() * (target != 0).bool()

#         target[thresh_mask] = 0

#         weight = num / torch.sum(target != 0)

#     loss = weight * F.cross_entropy(predict, target-1, weight=torch.FloatTensor([2, 1, 2, 2, 1, 1]).cuda(), ignore_index=-1)  # [10, 321, 321]

#     return loss



def contra_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    rep_teacher
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    # current_class_threshold = cfg["current_class_threshold"]
    # current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = 2, 6

    temp = 0.5
    num_queries = 256
    num_negatives = 512

    percent_low = 20
    percent_high = 80

    label_u_low  = copy.deepcopy(label_u)
    label_u_high  = copy.deepcopy(label_u)

    with torch.no_grad():

        entropy = -torch.sum(prob_u * torch.log(prob_u + 1e-10), dim=1)

        thresh_low = np.percentile(
            entropy[label_u_low != 0].detach().cpu().numpy().flatten(), percent_low
        )

        thresh_mask_low = entropy.ge(thresh_low).bool() * (label_u_low != 0).bool()

        label_u_low[thresh_mask_low] = 0


        thresh_high = np.percentile(
            entropy[label_u_high != 0].detach().cpu().numpy().flatten(), percent_high
        )
        thresh_mask_high = entropy.le(thresh_high).bool() * (label_u_high != 0).bool()

        label_u_high[thresh_mask_high] = 0


    label_l = label_onehot(label_l, 7)[:,1:,:,:]
    label_u_low = label_onehot(label_u_low, 7)[:,1:,:,:]
    label_u_high = label_onehot(label_u_high, 7)[:,1:,:,:]


    num_feat = rep.shape[1] #256
    num_labeled = label_l.shape[0]  #b
    num_segments = label_l.shape[1]  #6

    low_valid_pixel = torch.cat((label_l, label_u_low), dim=0)  # (2*num_labeled, num_cls, h, w)
    high_valid_pixel = torch.cat((label_l, label_u_high), dim=0) 

    rep = torch.cat((rep, rep), dim=0)
    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = torch.cat((rep_teacher, rep_teacher), dim=0)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_feat_negative_list = [] # candidate negative pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class


    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (2*num_labeled, num_cls, h, w)

    valid_classes = []

    for i in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        # rep_mask_low_entropy = (
        #     prob_seg > current_class_threshold
        # ) * low_valid_pixel_seg.bool()
        rep_mask_low_entropy = prob_seg * low_valid_pixel_seg.bool()

        # rep_mask_high_entropy = (
        #     prob_seg < current_class_negative_threshold
        # ) * high_valid_pixel_seg.bool()
        rep_mask_high_entropy = prob_seg * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy.bool()])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        negative_mask = rep_mask_high_entropy * class_mask
        seg_feat_negative_list.append(rep[negative_mask.bool()])


        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class

        return torch.tensor(0.0) * rep.sum()


    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
            ):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                if (
                    len(seg_feat_negative_list[i]) > 0
                ):
                    high_entropy_idx = torch.randint(
                        len(seg_feat_negative_list[i]), size=(num_queries * num_negatives,)
                    )
                    negative_feat = (
                        seg_feat_negative_list[i][high_entropy_idx].clone().cuda()
                    )

                    negative_feat = negative_feat.reshape(
                        num_queries, num_negatives, num_feat
                    )
                    positive_feat = (
                        seg_proto[i]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(num_queries, 1, 1)
                        .cuda()
                    )  # (num_queries, 1, num_feat)

                    all_feat = torch.cat(
                        (positive_feat, negative_feat), dim=1
                    )  # (num_queries, 1 + num_negative, num_feat)
                else:
                    reco_loss = reco_loss + 0 * rep.sum()
                    continue

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )

        return reco_loss / valid_seg



def get_criterion(cfg):
    cfg_criterion = cfg["criterion"]
    aux_weight = (
        cfg["net"]["aux_loss"]["loss_weight"]
        if cfg["net"].get("aux_loss", False)
        else 0
    )
    ignore_index = cfg["dataset"]["ignore_label"]
    if cfg_criterion["type"] == "ohem":
        criterion = CriterionOhem(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )
    else:
        criterion = Criterion(
            aux_weight, ignore_index=ignore_index, **cfg_criterion["kwargs"]
        )

    return criterion


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
