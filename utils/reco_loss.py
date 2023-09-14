# --------------------------------------------------------------------------------
# ReCo loss(without prob)
# --------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import copy

# def compute_reco_loss(rep, label, strong_threshold=0.5, temp=0.5, num_queries=512, num_negatives=512):
#     batch_size, num_feat, im_w_, im_h = rep.shape
#     num_segments = label.shape[1]

#     # compute valid binary mask for each pixel
#     valid_pixel = label

#     # permute representation for indexing: batch x im_h x im_w x feature_channel
#     rep = rep.permute(0, 2, 3, 1)

#     # compute prototype (class mean representation) for each class across all valid pixels
#     seg_feat_all_list = []
#     seg_feat_hard_list = []
#     seg_num_list = []
#     seg_proto_list = []
#     for i in range(num_segments):
#         valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
#         if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
#             continue

#         rep_mask_hard = valid_pixel_seg.bool()  # select hard queries
#         seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
#         seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
#         seg_feat_hard_list.append(rep[rep_mask_hard])
#         seg_num_list.append(int(valid_pixel_seg.sum().item()))


#     # compute regional contrastive loss
#     if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         return torch.tensor(0.0)
#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)
#         valid_seg = len(seg_num_list)
#         seg_len = torch.arange(valid_seg)

#         for i in range(valid_seg):
#             # sample hard queries
#             if len(seg_feat_hard_list[i]) > 0:
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
#                 anchor_feat = anchor_feat_hard
#             else:  # in some rare cases, all queries in the current query class are easy
#                 continue

#             # apply negative key sampling (with no gradients)
#             with torch.no_grad():
#                 # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
#                 seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

#                 # compute similarity for each negative segment prototype (semantic class relation graph)
#                 proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
#                 proto_prob = torch.softmax(proto_sim / temp, dim=0)

#                 # sampling negative keys based on the generated distribution [num_queries x num_negatives]
#                 negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
#                 samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
#                 samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

#                 # sample negative indices from each negative class
#                 negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
#                 negative_index = negative_index_sampler(samp_num, negative_num_list)

#                 # index negative keys (from other classes)
#                 negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
#                 negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

#                 # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
#                 positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
#                 all_feat = torch.cat((positive_feat, negative_feat), dim=1)

#             seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
#             reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
#         return reco_loss / valid_seg


# def negative_index_sampler(samp_num, seg_num_list):
#     negative_index = []
#     for i in range(samp_num.shape[0]):
#         for j in range(samp_num.shape[1]):
#             negative_index += np.random.randint(low=sum(seg_num_list[:j]),
#                                                 high=sum(seg_num_list[:j+1]),
#                                                 size=int(samp_num[i, j])).tolist()
#     return negative_index


# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import copy

# def compute_reco_loss(rep, label, strong_threshold=1.0, temp=0.5, num_queries=512, num_negatives=512):
#     batch_size, num_feat, im_w_, im_h = rep.shape
#     num_segments = label.shape[1]

#     # compute valid binary mask for each pixel
#     valid_pixel = label

#     # permute representation for indexing: batch x im_h x im_w x feature_channel
#     rep = rep.permute(0, 2, 3, 1)

#     # compute prototype (class mean representation) for each class across all valid pixels
#     seg_feat_all_list = []
#     seg_feat_hard_list = []
#     seg_feat_easy_list = []
#     seg_num_list = []
#     seg_proto_list = []
#     seg_negative_all_list = []
#     seg_negative_hard_list = []
#     seg_negative_easy_list = []
#     for i in range(num_segments):
#         valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
#         valid_pixel_negative = torch.ones_like(valid_pixel[:, i]) - valid_pixel[:, i]
#         if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
#             continue

#         # prob_seg = prob[:, i, :, :]
#         # rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
#         # rep_mask_easy = (prob_seg >= strong_threshold) * valid_pixel_seg.bool()

#         # rep_negative_hard = (prob_seg > (1-strong_threshold)) * valid_pixel_negative.bool()  # select hard queries
#         # rep_negative_easy = (prob_seg <= (1-strong_threshold)) * valid_pixel_negative.bool()

#         seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
#         seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
#         # seg_feat_hard_list.append(rep[rep_mask_hard])
#         # seg_feat_easy_list.append(rep[rep_mask_easy])
#         seg_num_list.append(int(valid_pixel_seg.sum().item()))
#         seg_negative_all_list.append(rep[valid_pixel_negative.bool()])
#         # seg_negative_hard_list.append(rep[rep_negative_hard.bool()])
#         # seg_negative_easy_list.append(rep[rep_negative_easy.bool()])


#     # compute regional contrastive loss
#     if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         return torch.tensor(0.0)
#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)
#         valid_seg = len(seg_num_list)

#         for i in range(valid_seg):
#             # sample hard queries
#             if len(seg_feat_all_list[i]) > 0:
#                 #anchor的灵活抽样策略
#                 seg_all_idx = torch.randint(len(seg_feat_all_list[i]), size=(num_queries,))
#                 anchor_feat_all = seg_feat_all_list[i][seg_all_idx]
#                 anchor_feat = anchor_feat_all

#             else:  # in some rare cases, all queries in the current query class are easy
#                 continue

#             # apply negative key sampling (with no gradients)
#             with torch.no_grad():

#                 # seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives,))
#                 # negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]
#                 # negative_feat = negative_feat_hard
#                 # negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)
#                 seg_negative_idx = torch.randint(len(seg_negative_all_list[i]), size=(num_queries*num_negatives,))
#                 negative_feat = seg_negative_all_list[i][seg_negative_idx]
#                 negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

#                 # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
#                 positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
#                 all_feat = torch.cat((positive_feat, negative_feat), dim=1)

#             seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
#             reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
#         return reco_loss / valid_seg


# --------------------------------------------------------------------------------
# ReCo loss(with prob)
# --------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import copy

# def compute_reco_loss(rep, label, prob, strong_threshold=0.97, temp=0.5, num_queries=512, num_negatives=512):
#     batch_size, num_feat, im_w_, im_h = rep.shape
#     num_segments = label.shape[1]

#     # compute valid binary mask for each pixel
#     valid_pixel = label

#     # permute representation for indexing: batch x im_h x im_w x feature_channel
#     rep = rep.permute(0, 2, 3, 1)

#     # compute prototype (class mean representation) for each class across all valid pixels
#     seg_feat_all_list = []
#     seg_feat_hard_list = []
#     seg_num_list = []
#     seg_proto_list = []
#     for i in range(num_segments):
#         valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
#         if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
#             continue

#         prob_seg = prob[:, i, :, :]
#         rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
#         seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
#         seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
#         seg_feat_hard_list.append(rep[rep_mask_hard])
#         seg_num_list.append(int(valid_pixel_seg.sum().item()))


#     # compute regional contrastive loss
#     if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         return torch.tensor(0.0)
#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)
#         valid_seg = len(seg_num_list)
#         seg_len = torch.arange(valid_seg)

#         for i in range(valid_seg):
#             # sample hard queries
#             if len(seg_feat_hard_list[i]) > 0:
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
#                 anchor_feat = anchor_feat_hard
#             else:  # in some rare cases, all queries in the current query class are easy
#                 continue

#             # apply negative key sampling (with no gradients)
#             with torch.no_grad():
#                 # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
#                 seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

#                 # compute similarity for each negative segment prototype (semantic class relation graph)
#                 proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
#                 proto_prob = torch.softmax(proto_sim / temp, dim=0)

#                 # sampling negative keys based on the generated distribution [num_queries x num_negatives]
#                 negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
#                 samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
#                 samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

#                 # sample negative indices from each negative class
#                 negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
#                 negative_index = negative_index_sampler(samp_num, negative_num_list)

#                 # index negative keys (from other classes)
#                 negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
#                 negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

#                 # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
#                 positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
#                 all_feat = torch.cat((positive_feat, negative_feat), dim=1)

#             seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
#             reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
#         return reco_loss / valid_seg


# def negative_index_sampler(samp_num, seg_num_list):
#     negative_index = []
#     for i in range(samp_num.shape[0]):
#         for j in range(samp_num.shape[1]):
#             negative_index += np.random.randint(low=sum(seg_num_list[:j]),
#                                                 high=sum(seg_num_list[:j+1]),
#                                                 size=int(samp_num[i, j])).tolist()
#     return negative_index



# --------------------------------------------------------------------------------
# ReCo loss(with prob)
# --------------------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import copy

# def compute_reco_loss(rep, label, prob1, prob2, strong_threshold, weak_threshold, temp=0.5, num_queries=512, num_negatives=512):
#     batch_size, num_feat, im_w_, im_h = rep.shape
#     num_segments = label.shape[1]

#     # compute valid binary mask for each pixel
#     valid_pixel = label

#     # permute representation for indexing: batch x im_h x im_w x feature_channel
#     rep = rep.permute(0, 2, 3, 1)

#     # compute prototype (class mean representation) for each class across all valid pixels
#     seg_feat_all_list = []
#     seg_feat_hard_list = []
#     seg_feat_easy_list = []
#     seg_num_list = []
#     seg_proto_list = []
#     seg_negative_all_list = []
#     seg_negative_hard_list = []
#     seg_negative_easy_list = []
#     for i in range(num_segments):
#         valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
#         valid_pixel_negative = torch.ones_like(valid_pixel[:, i]) - valid_pixel[:, i]
#         if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
#             continue

#         prob_seg1 = prob1[:, i, :, :]
#         prob_seg2 = prob2[:, i, :, :]
#         rep_mask_hard = ((prob_seg1 < weak_threshold)|(prob_seg2 < weak_threshold)) * valid_pixel_seg.bool()  # select hard queries
#         rep_mask_easy = ((prob_seg1 > strong_threshold)|(prob_seg2 > strong_threshold)) * valid_pixel_seg.bool()

#         rep_negative_hard = ((prob_seg1 < weak_threshold)|(prob_seg2 < weak_threshold)) * valid_pixel_negative.bool()  # select hard queries
#         rep_negative_easy = ((prob_seg1 > strong_threshold)|(prob_seg2 > strong_threshold)) * valid_pixel_negative.bool()

#         seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
#         seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
#         seg_feat_hard_list.append(rep[rep_mask_hard])
#         seg_feat_easy_list.append(rep[rep_mask_easy])
#         seg_num_list.append(int(valid_pixel_seg.sum().item()))
#         seg_negative_all_list.append(rep[valid_pixel_negative.bool()])
#         seg_negative_hard_list.append(rep[rep_negative_hard.bool()])
#         seg_negative_easy_list.append(rep[rep_negative_easy.bool()])


#     # compute regional contrastive loss
#     if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         return torch.tensor(0.0)
#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)
#         valid_seg = len(seg_num_list)

#         for i in range(valid_seg):
#             # sample hard queries
#             if len(seg_feat_easy_list[i]) > 0:
#                 #anchor的灵活抽样策略
#                 seg_easy_idx = torch.randint(len(seg_feat_easy_list[i]), size=(num_queries//2,))
#                 anchor_feat_easy = seg_feat_easy_list[i][seg_easy_idx]
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries//2,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]       
#                 anchor_feat = torch.cat((anchor_feat_hard,anchor_feat_easy))
#             elif len(seg_feat_easy_list[i]) == 0:
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]       
#                 anchor_feat = anchor_feat_hard 

#             else:  # in some rare cases, all queries in the current query class are easy
#                 continue

#             # apply negative key sampling (with no gradients)
#             with torch.no_grad():

#                 # seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives,))
#                 # negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]
#                 # negative_feat = negative_feat_hard
#                 # negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)
#                 # seg_negative_idx = torch.randint(len(seg_negative_all_list[i]), size=(num_queries*num_negatives,))
#                 # negative_feat = seg_negative_all_list[i][seg_negative_idx]
#                 # negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)


#                 if len(seg_negative_easy_list[i]) > 0:
#                     #anchor的灵活抽样策略
#                     seg_negative_easy_idx = torch.randint(len(seg_negative_easy_list[i]), size=(num_queries*num_negatives//2,))
#                     negative_feat_easy = seg_negative_easy_list[i][seg_negative_easy_idx]
#                     seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives//2,))
#                     negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]       
#                     negative_feat = torch.cat((negative_feat_hard, negative_feat_easy))
#                 elif len(seg_negative_easy_list[i]) == 0:
#                     seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives,))
#                     negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]       
#                     negative_feat = negative_feat_hard 
#                 negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

#                 # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
#                 positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
#                 all_feat = torch.cat((positive_feat, negative_feat), dim=1)

#             seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
#             reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
#         return reco_loss / valid_seg



# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import copy

# def compute_reco_loss(rep, label, prob1, prob2, epoch, strong_threshold_change, weak_threshold_change, strong_threshold_unchange, weak_threshold_unchange, temp=0.5, num_queries=512, num_negatives=512):
#     batch_size, num_feat, im_w_, im_h = rep.shape
#     num_segments = label.shape[1]

#     # compute valid binary mask for each pixel
#     valid_pixel = label

#     # permute representation for indexing: batch x im_h x im_w x feature_channel
#     rep = rep.permute(0, 2, 3, 1)

#     # compute prototype (class mean representation) for each class across all valid pixels
#     seg_feat_all_list = []
#     seg_feat_hard_list = []
#     seg_feat_easy_list = []
#     seg_num_list = []
#     seg_proto_list = []
#     seg_negative_all_list = []
#     seg_negative_hard_list = []
#     seg_negative_easy_list = []

#     # threshold_change = weak_threshold_change + (strong_threshold_change - weak_threshold_change) * epoch / 29
#     # threshold_unchange = weak_threshold_unchange + (strong_threshold_unchange - weak_threshold_unchange) * epoch / 29
#     # threshold = [threshold_unchange, threshold_change]
#     threshold_change = strong_threshold_change
#     threshold_unchange = strong_threshold_unchange
#     threshold = [threshold_unchange, threshold_change]

#     for i in range(num_segments):
#         valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
#         valid_pixel_negative = torch.ones_like(valid_pixel[:, i]) - valid_pixel[:, i]
#         if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
#             continue

#         prob_seg1 = prob1[:, i, :, :]
#         prob_seg2 = prob2[:, i, :, :]
#         rep_mask_hard = ((prob_seg1 <= threshold[i])|(prob_seg2 <= threshold[i])) * valid_pixel_seg.bool()  # select hard queries
#         rep_mask_easy = ((prob_seg1 > threshold[i])&(prob_seg2 > threshold[i])) * valid_pixel_seg.bool()

#         rep_negative_hard = ((prob_seg1 <= threshold[i])|(prob_seg2 <= threshold[i])) * valid_pixel_negative.bool()  # select hard queries
#         rep_negative_easy = ((prob_seg1 > threshold[i])&(prob_seg2 > threshold[i])) * valid_pixel_negative.bool()

#         seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
#         seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
#         seg_feat_hard_list.append(rep[rep_mask_hard])
#         seg_feat_easy_list.append(rep[rep_mask_easy])
#         seg_num_list.append(int(valid_pixel_seg.sum().item()))
#         seg_negative_all_list.append(rep[valid_pixel_negative.bool()])
#         seg_negative_hard_list.append(rep[rep_negative_hard.bool()])
#         seg_negative_easy_list.append(rep[rep_negative_easy.bool()])



#     num_queries_hard=int((256+(512-256)*epoch/29))
#     num_queries_easy=512-num_queries_hard
#     num_negative_hard=int((131072+(262144-131072)*epoch/29))
#     num_negative_easy=262144-num_negative_hard



#     # compute regional contrastive loss
#     if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         return torch.tensor(0.0)
#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)
#         valid_seg = len(seg_num_list)

#         for i in range(valid_seg):
#             # sample hard queries
#             if len(seg_feat_easy_list[i]) > 0:
#                 #anchor的灵活抽样策略
#                 seg_easy_idx = torch.randint(len(seg_feat_easy_list[i]), size=(num_queries_easy,))
#                 anchor_feat_easy = seg_feat_easy_list[i][seg_easy_idx]
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries_hard,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]       
#                 anchor_feat = torch.cat((anchor_feat_hard,anchor_feat_easy))
#             elif len(seg_feat_easy_list[i]) == 0:
#                 seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
#                 anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]       
#                 anchor_feat = anchor_feat_hard 

#             else:  # in some rare cases, all queries in the current query class are easy
#                 continue

#             # apply negative key sampling (with no gradients)
#             with torch.no_grad():

#                 # seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives,))
#                 # negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]
#                 # negative_feat = negative_feat_hard
#                 # negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)
#                 # seg_negative_idx = torch.randint(len(seg_negative_all_list[i]), size=(num_queries*num_negatives,))
#                 # negative_feat = seg_negative_all_list[i][seg_negative_idx]
#                 # negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)


#                 if len(seg_negative_easy_list[i]) > 0:
#                     #anchor的灵活抽样策略
#                     seg_negative_easy_idx = torch.randint(len(seg_negative_easy_list[i]), size=(num_negative_easy,))
#                     negative_feat_easy = seg_negative_easy_list[i][seg_negative_easy_idx]
#                     seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_negative_hard,))
#                     negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]       
#                     negative_feat = torch.cat((negative_feat_hard, negative_feat_easy))
#                 elif len(seg_negative_easy_list[i]) == 0:
#                     seg_negative_hard_idx = torch.randint(len(seg_negative_hard_list[i]), size=(num_queries*num_negatives,))
#                     negative_feat_hard = seg_negative_hard_list[i][seg_negative_hard_idx]       
#                     negative_feat = negative_feat_hard 
#                 negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

#                 # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
#                 positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
#                 all_feat = torch.cat((positive_feat, negative_feat), dim=1)

#             seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
#             reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
#         return reco_loss / valid_seg



import torch
import torch.nn.functional as F
import numpy as np
import os
import copy

def compute_reco_loss(rep, label, prob1, prob2, epoch, strong_threshold_change, weak_threshold_change, strong_threshold_unchange, weak_threshold_unchange, temp=0.5, num_queries=512, num_negatives=512):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]

    # compute valid binary mask for each pixel
    valid_pixel = label

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_feat_easy_list = []
    seg_num_list = []
    seg_proto_list = []
    seg_negative_all_list = []
    seg_negative_hard_list = []
    seg_negative_easy_list = []

    # threshold_change = weak_threshold_change + (strong_threshold_change - weak_threshold_change) * epoch / 29
    # threshold_unchange = weak_threshold_unchange + (strong_threshold_unchange - weak_threshold_unchange) * epoch / 29
    # threshold = [threshold_unchange, threshold_change]
    threshold_change = strong_threshold_change
    threshold_unchange = strong_threshold_unchange
    threshold = [threshold_unchange, threshold_change]

    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        valid_pixel_negative = torch.ones_like(valid_pixel[:, i]) - valid_pixel[:, i]
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg1 = prob1[:, i, :, :]
        prob_seg2 = prob2[:, i, :, :]
        rep_mask_hard = ((prob_seg1 <= threshold[i])|(prob_seg2 <= threshold[i])) * valid_pixel_seg.bool()  # select hard queries
        rep_mask_easy = ((prob_seg1 > threshold[i])&(prob_seg2 > threshold[i])) * valid_pixel_seg.bool()

        rep_negative_hard = ((prob_seg1 <= threshold[i])|(prob_seg2 <= threshold[i])) * valid_pixel_negative.bool()  # select hard queries
        rep_negative_easy = ((prob_seg1 > threshold[i])&(prob_seg2 > threshold[i])) * valid_pixel_negative.bool()

        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_feat_easy_list.append(rep[rep_mask_easy])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))
        seg_negative_all_list.append(rep[valid_pixel_negative.bool()])
        seg_negative_hard_list.append(rep[rep_negative_hard.bool()])
        seg_negative_easy_list.append(rep[rep_negative_easy.bool()])


    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_all_list[i]) > 0:
                #anchor的灵活抽样策略
                seg_all_idx = torch.randint(len(seg_feat_all_list[i]), size=(512,))
                anchor_feat = seg_feat_all_list[i][seg_all_idx]
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():

                seg_negative_all_idx = torch.randint(len(seg_negative_all_list[i]), size=(num_queries*num_negatives,))
                negative_feat = seg_negative_all_list[i][seg_negative_all_idx]
                negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
        return reco_loss / valid_seg

















