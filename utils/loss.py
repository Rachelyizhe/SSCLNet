import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr
    
    return queue, queue_ptr


def compute_contra_memobank_loss(
    rep_all,
    prob_all,
    label_all,
    mask_all,
    anchor_memobank,
    anchor_queue_ptrlis,
    anchor_queue_size,
    prototype,
    negative_memobank,
    negative_queue_ptrlis,
    negative_queue_size, 
):

    class_anchor_threshold = 0.3
    low_rank = 2
    temp = 0.5 
    num_anchor = 256 # num_queries_anchor
    num_negatives = 50 # num_negatives
    num_feat = rep_all.shape[1] #256
    num_segments = label_all.shape[1] #6

    rep_all = rep_all.permute(0, 2, 3, 1) #torch.Size([16, 128, 128, 6])
    _, prob_indices = torch.sort(prob_all, 1, True)
    prob_indices = prob_indices.permute(0, 2, 3, 1)  #torch.Size([16, 128, 128, 6])

    seg_feat_positive_list = []
    seg_feat_positive_mean_list = []
    seg_feat_anchor_list = []  # candidate anchor pixels
    seg_feat_negative_list = [] 
    seg_feat_positive_list_num = []
    seg_feat_anchor_list_num = []  
    seg_feat_negative_list_num = [] 


    for i in range(num_segments):

        # generate positive mask
        positive_mask = label_all[:, i, :, :]  # select binary mask for i-th class   torch.Size([16, 128, 128])

        # generate anchor mask
        prob_anchor = prob_all[:, i, :, :] # torch.Size([16, 128, 128])
        anchor_mask = (
            prob_anchor > class_anchor_threshold
        ) * positive_mask.bool() # torch.Size([16, 128, 128])
        # generate negative mask
        class_mask = torch.sum(prob_indices[:, :, :, :low_rank].eq(i), dim=3).bool() # torch.Size([16, 128, 128])
        negative_mask = class_mask * (label_all[:, i] == 0).bool() # torch.Size([16, 128, 128])
        negative_mask = negative_mask * ((mask_all != 0).bool()).bool() # torch.Size([16, 128, 128])

        seg_feat_positive_list.append(rep_all[positive_mask.bool()])  
        if positive_mask.sum() > 0:
            seg_feat_positive_mean_list.append(
                torch.mean(
                    rep_all[positive_mask.bool()], dim=0, keepdim=True
                    )
            )  
        else:
            seg_feat_positive_mean_list.append(torch.zeros(1, 256).cuda())
        seg_feat_anchor_list.append(rep_all[anchor_mask])
        seg_feat_negative_list.append(rep_all[negative_mask])  
        seg_feat_positive_list_num.append(positive_mask.sum().item()) 
        #[992, 30537, 26677, 12521, 17501, 120]
        #
        #
        seg_feat_anchor_list_num.append(anchor_mask.sum().item())
        #[458, 1932, 9977, 3059, 7071, 0]
        #
        #
        seg_feat_negative_list_num.append(negative_mask.sum().item()) 
        #[57741, 3000, 32714, 24240, 24199, 6727]
        #
        #

    valid_seg = 0
    reco_loss = torch.tensor(0.0).cuda()

    for i in range(num_segments):
        # choose anchor feat
        if(0 < seg_feat_anchor_list_num[i] < num_anchor):
            if(anchor_memobank[i][0].shape[0] > 0):
                seg_feat_anchor_memobank_idx = torch.randint(
                    anchor_memobank[i][0].shape[0], size=(num_anchor - seg_feat_anchor_list_num[i],)
                )
                anchor_feat_memobank = (
                    anchor_memobank[i][0][seg_feat_anchor_memobank_idx].clone().cuda()
                )
                anchor_feat = torch.cat((seg_feat_anchor_list[i].clone().cuda(), anchor_feat_memobank), dim = 0)
            else:
                seg_feat_anchor_idx = torch.randint(
                    seg_feat_anchor_list_num[i], size=(num_anchor,)
                )              
                anchor_feat = (
                    seg_feat_anchor_list[i][seg_feat_anchor_idx].clone().cuda()
                )
        elif(seg_feat_anchor_list_num[i] >= num_anchor):
            seg_feat_anchor_idx = torch.randint(
                seg_feat_anchor_list_num[i], size=(num_anchor,)
            )              
            anchor_feat = (
                seg_feat_anchor_list[i][seg_feat_anchor_idx].clone().cuda()
            )      
        elif(seg_feat_anchor_list_num[i] == 0):
            if(seg_feat_positive_list_num[i] > 0):
                seg_feat_positive_idx = torch.randint(
                    seg_feat_positive_list_num[i], size=(num_anchor,)
                )              
                anchor_feat = (
                    seg_feat_positive_list[i][seg_feat_positive_idx].clone().cuda()
                )    
            elif(anchor_memobank[i][0].shape[0] > 0):
                seg_feat_anchor_memobank_idx = torch.randint(
                    anchor_memobank[i][0].shape[0], size=(num_anchor,)
                )
                anchor_feat = (
                    anchor_memobank[i][0][seg_feat_anchor_memobank_idx].clone().cuda()
                )
            else:
                reco_loss = reco_loss + 0 * rep_all.sum()
                continue
        
        # compute positive feat
        if(seg_feat_positive_list_num[i] > 0):
            positive_feat = seg_feat_positive_mean_list[i]
            if not (prototype[i] == 0).all():
                ema_decay = 0.99
                positive_feat = (
                    1 - ema_decay
                ) * positive_feat + ema_decay * prototype[i]
            positive_feat = (
                positive_feat
                .unsqueeze(0)
                .repeat(num_anchor, 1, 1)
                .cuda()
            )  # (num_anchor, 1, num_feat)
        else:
            if not (prototype[i] == 0).all():
                positive_feat =  prototype[i]
                positive_feat = (
                    positive_feat
                    .unsqueeze(0)
                    .repeat(num_anchor, 1, 1)
                    .cuda()
                )  # (num_anchor, 1, num_feat)         
            else:
                reco_loss = reco_loss + 0 * rep_all.sum()
                continue

        # choose negative feat
        if(0 <= seg_feat_negative_list_num[i] < num_anchor * num_negatives):
            if(negative_memobank[i][0].shape[0] > 0):
                seg_feat_negative_memobank_idx = torch.randint(
                    negative_memobank[i][0].shape[0], size=(num_anchor * num_negatives - seg_feat_negative_list_num[i],)
                )
                negative_feat_memobank = (
                    negative_memobank[i][0][seg_feat_negative_memobank_idx].clone().cuda()
                )
                negative_feat = torch.cat((seg_feat_negative_list[i].clone().cuda(), negative_feat_memobank), dim = 0)
                negative_feat = negative_feat.reshape(
                    num_anchor, num_negatives, num_feat
                )
            else:
                if seg_feat_negative_list_num[i] > 0:
                    seg_feat_negative_idx = torch.randint(
                        seg_feat_negative_list_num[i], size=(num_anchor * num_negatives,)
                    )              
                    negative_feat = (
                        seg_feat_negative_list[i][seg_feat_negative_idx].clone().cuda()
                    )
                    negative_feat = negative_feat.reshape(
                    num_anchor, num_negatives, num_feat
                    )
                else:
                    reco_loss = reco_loss + 0 * rep_all.sum()
                    continue

        else:
            seg_feat_negative_idx = torch.randint(
                seg_feat_negative_list_num[i], size=(num_anchor * num_negatives,)
            )              
            negative_feat = (
                seg_feat_negative_list[i][seg_feat_negative_idx].clone().cuda()
            )      
            negative_feat = negative_feat.reshape(
            num_anchor, num_negatives, num_feat
            )

        all_feat = torch.cat(
            (positive_feat, negative_feat), dim=1
        )  # (num_anchor, 1 + num_negative, num_feat)

        seg_logits = torch.cosine_similarity(
            anchor_feat.unsqueeze(1), all_feat, dim=2
        )

        reco_loss = reco_loss + F.cross_entropy(
            seg_logits / temp, torch.zeros(num_anchor).long().cuda()
        )
        valid_seg = valid_seg + 1


    for i in range(num_segments):
        # anchor dequeue_and_enqueue
        if seg_feat_anchor_list[i].shape[0] > 0:
            anchor_memobank[i], anchor_queue_ptrlis[i] = dequeue_and_enqueue(
                keys = seg_feat_anchor_list[i].detach(),
                queue = anchor_memobank[i],
                queue_ptr = anchor_queue_ptrlis[i],
                queue_size = anchor_queue_size[i],
            )
        # positive-mean dequeue_and_enqueue
        if seg_feat_positive_list_num[i] > 0:
            positive_feat = seg_feat_positive_mean_list[i].detach()
            if not (prototype[i] == 0).all():
                ema_decay = 0.99
                prototype[i] = (
                    1 - ema_decay
                ) * positive_feat + ema_decay * prototype[i]
            else:
                prototype[i] = positive_feat
        # negative dequeue_and_enqueue
        if seg_feat_negative_list[i].shape[0] > 0:
            negative_memobank[i], negative_queue_ptrlis[i] = dequeue_and_enqueue(
                keys = seg_feat_negative_list[i].detach(),
                queue = negative_memobank[i],
                queue_ptr = negative_queue_ptrlis[i],
                queue_size = negative_queue_size[i],
            )


    return  anchor_memobank, anchor_queue_ptrlis, prototype, \
        negative_memobank, negative_queue_ptrlis, reco_loss / valid_seg








# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from .utils import dequeue_and_enqueue


# def compute_contra_memobank_loss(
#     rep_all,
#     prob_all,
#     label_all,
#     mask_all,
#     memobank,
#     queue_ptrlis,
#     queue_size,
#     momentum_prototype=None,
    
# ):

#     # class_anchor_threshold: delta_p (0.3)
#     class_anchor_threshold = 0.3
#     low_rank = 2 
#     temp = 0.5 
#     num_queries = 256 # num_queries_anchor
#     num_negatives = 50 # num_negatives

#     num_feat = rep_all.shape[1] #256
#     num_segments = label_all.shape[1] #6

#     rep_all = rep_all.permute(0, 2, 3, 1) #torch.Size([16, 128, 128, 6])

#     seg_feat_all_list = []
#     seg_feat_anchor_list = []  # candidate anchor pixels
#     seg_num_list = []  # the number of pixels in each class
#     seg_proto_list = []  # the center of each class

#     _, prob_indices = torch.sort(prob_all, 1, True)
#     prob_indices = prob_indices.permute(0, 2, 3, 1)  #torch.Size([16, 128, 128, 6])

#     valid_classes = []
#     new_keys = []


#     for i in range(num_segments):

#         # generate positive mask
#         valid_pixel_seg = label_all[:, i, :, :]  # select binary mask for i-th class   torch.Size([16, 128, 128])
#         prob_seg = prob_all[:, i, :, :] # torch.Size([16, 128, 128])

#         # generate anchor mask
#         rep_mask_anchor = (
#             prob_seg > class_anchor_threshold
#         ) * valid_pixel_seg.bool() # torch.Size([16, 128, 128])

#         seg_feat_all_list.append(rep_all[valid_pixel_seg.bool()])  
#         '''
#         i=0,seg_feat_all_list[0].size() = torch.Size([668, 256])
#         i=1,seg_feat_all_list[1].size() = torch.Size([18654, 256])
#         i=2,seg_feat_all_list[2].size() = torch.Size([10333, 256])
#         i=3,seg_feat_all_list[3].size() = torch.Size([1857, 256])
#         i=4,seg_feat_all_list[4].size() = torch.Size([15854, 256])
#         i=5,seg_feat_all_list[5].size() = torch.Size([0, 256])
#         '''
#         seg_feat_anchor_list.append(rep_all[rep_mask_anchor]) 
#         '''
#         i=0,seg_feat_anchor_list[0].size() = torch.Size([129, 256])
#         i=1,seg_feat_anchor_list[1].size() = torch.Size([1294, 256])
#         i=2,seg_feat_anchor_list[2].size() = torch.Size([2461, 256])
#         i=3,seg_feat_anchor_list[3].size() = torch.Size([811, 256])
#         i=4,seg_feat_anchor_list[4].size() = torch.Size([2660, 256])
#         i=5,seg_feat_anchor_list[5].size() = torch.Size([0, 256])
#         '''
#         # positive sample: center of the class
#         seg_proto_list.append(
#             torch.mean(
#                 rep_all[valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
#             )# torch.Size([1, 256])
#         )
        
#         # generate negative mask
#         class_mask = torch.sum(prob_indices[:, :, :, :low_rank].eq(i), dim=3).bool() # torch.Size([16, 128, 128])
#         negative_mask = class_mask * (label_all[:, i] == 0).bool() # torch.Size([16, 128, 128])
#         negative_mask = negative_mask * ((mask_all != 0).bool()).bool() # torch.Size([16, 128, 128])


#         keys = rep_all[negative_mask].detach()
#         '''
#         i=0,keys.size() = torch.Size([18392, 256])
#         i=1,keys.size() = torch.Size([4273, 256])
#         i=2,keys.size() = torch.Size([13644, 256])
#         i=3,keys.size() = torch.Size([26809, 256])
#         i=4,keys.size() = torch.Size([10115, 256])
#         i=5,keys.size() = torch.Size([11119, 256])
#         '''

#         new_keys.append(
#             dequeue_and_enqueue(
#                 keys=keys,
#                 queue=memobank[i],
#                 queue_ptr=queue_ptrlis[i],
#                 queue_size=queue_size[i],
#             )
#         )

#         if valid_pixel_seg.sum() > 0:
#             seg_num_list.append(int(valid_pixel_seg.sum().item())) # [668, 18654, 10333, 1857, 15854]
#             valid_classes.append(i) # [0, 1, 2, 3, 4]
# #------------------------------------------------------------------------------


#     reco_loss = torch.tensor(0.0).cuda()
#     seg_proto = torch.cat(seg_proto_list)  # shape: [valid_classes, 256]
#     valid_seg = len(seg_num_list)  # number of valid classes

#     prototype = torch.zeros(
#         (prob_indices.shape[-1], num_queries, 1, num_feat)
#     ).cuda() # torch.Size([6, 256, 1, 256])

#     for i in range(valid_seg):
#         if (
#             #len(seg_feat_anchor_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0
#             memobank[valid_classes[i]][0].shape[0] > 0
#         ):
#             # select anchor pixel
#             # seg_feat_anchor_idx = torch.randint(
#             #     len(seg_feat_anchor_list[i]), size=(num_queries,)
#             # )
#             # anchor_feat = (
#             #     seg_feat_anchor_list[i][seg_feat_anchor_idx].clone().cuda()
#             # )
#             seg_feat_anchor_idx = torch.randint(
#                 len(seg_feat_all_list[valid_classes[i]]), size=(num_queries,)
#             )
#             anchor_feat = (
#                 seg_feat_all_list[valid_classes[i]][seg_feat_anchor_idx].clone().cuda()
#             )
#         else:
#             # in some rare cases, all queries in the current query class are easy
#             reco_loss = reco_loss + 0 * rep_all.sum()
#             continue

#         # apply negative key sampling from memory bank (with no gradients)
#         with torch.no_grad():
#             negative_feat = memobank[valid_classes[i]][0].clone().cuda()

#             negative_feat_idx = torch.randint(
#                 len(negative_feat), size=(num_queries * num_negatives,)
#             )
#             negative_feat = negative_feat[negative_feat_idx]
#             negative_feat = negative_feat.reshape(
#                 num_queries, num_negatives, num_feat
#             )
#             positive_feat = (
#                 seg_proto[valid_classes[i]]
#                 .unsqueeze(0)
#                 .unsqueeze(0)
#                 .repeat(num_queries, 1, 1)
#                 .cuda()
#             )  # (num_queries, 1, num_feat)

#             if momentum_prototype is not None:
#                 # if not (momentum_prototype == 0).all():
#                 #     ema_decay = min(1 - 1 / i_iter, 0.999)
#                 #     positive_feat = (
#                 #         1 - ema_decay
#                 #     ) * positive_feat + ema_decay * momentum_prototype[
#                 #         valid_classes[i]
#                 #     ]
#                 prototype[valid_classes[i]] = positive_feat.clone()

#             all_feat = torch.cat(
#                 (positive_feat, negative_feat), dim=1
#             )  # (num_queries, 1 + num_negative, num_feat)

#         seg_logits = torch.cosine_similarity(
#             anchor_feat.unsqueeze(1), all_feat, dim=2
#         )

#         reco_loss = reco_loss + F.cross_entropy(
#             seg_logits / temp, torch.zeros(num_queries).long().cuda()
#         )

#     return prototype, reco_loss / valid_seg







# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from .utils import dequeue_and_enqueue


# def compute_contra_memobank_loss(
#     rep_all,
#     prob_all,
#     label_all,
#     mask_all,
#     memobank,
#     queue_ptrlis,
#     queue_size,
#     momentum_prototype=None,
# ):

#     # class_anchor_threshold: delta_p (0.3)
#     class_anchor_threshold = 0.3
#     low_rank = 2 
#     temp = 0.5 
#     num_queries = 256 # num_queries_anchor
#     num_negatives = 50 # num_negatives

#     num_feat = rep_all.shape[1] #256
#     num_segments = label_all.shape[1] #6

#     rep_all = rep_all.permute(0, 2, 3, 1) #torch.Size([16, 128, 128, 6])

#     seg_feat_all_list = []
#     # seg_feat_anchor_list = []  # candidate anchor pixels
#     seg_num_list = []  # the number of pixels in each class
#     seg_proto_list = []  # the center of each class

#     _, prob_indices = torch.sort(prob_all, 1, True)
#     prob_indices = prob_indices.permute(0, 2, 3, 1)  #torch.Size([16, 128, 128, 6])

#     valid_classes = []
#     new_keys = []


#     for i in range(num_segments):

#         valid_pixel_seg = label_all[:, i, :, :]  # select binary mask for i-th class   torch.Size([16, 128, 128])
#         # prob_seg = prob_all[:, i, :, :] # torch.Size([16, 128, 128])
#         # rep_mask_anchor = (
#         #     prob_seg > class_anchor_threshold
#         # ) * valid_pixel_seg.bool() # torch.Size([16, 128, 128])

#         seg_feat_all_list.append(rep_all[valid_pixel_seg.bool()])  
#         '''
#         i=0,seg_feat_all_list[0].size() = torch.Size([668, 256])
#         i=1,seg_feat_all_list[1].size() = torch.Size([18654, 256])
#         i=2,seg_feat_all_list[2].size() = torch.Size([10333, 256])
#         i=3,seg_feat_all_list[3].size() = torch.Size([1857, 256])
#         i=4,seg_feat_all_list[4].size() = torch.Size([15854, 256])
#         i=5,seg_feat_all_list[5].size() = torch.Size([0, 256])
#         '''
#         # seg_feat_anchor_list.append(rep_all[rep_mask_anchor]) 
#         '''
#         i=0,seg_feat_anchor_list[0].size() = torch.Size([129, 256])
#         i=1,seg_feat_anchor_list[1].size() = torch.Size([1294, 256])
#         i=2,seg_feat_anchor_list[2].size() = torch.Size([2461, 256])
#         i=3,seg_feat_anchor_list[3].size() = torch.Size([811, 256])
#         i=4,seg_feat_anchor_list[4].size() = torch.Size([2660, 256])
#         i=5,seg_feat_anchor_list[5].size() = torch.Size([0, 256])
#         '''
#         # positive sample: center of the class
#         seg_proto_list.append(
#             torch.mean(
#                 rep_all[valid_pixel_seg.bool()], dim=0, keepdim=True
#             )# torch.Size([1, 256])
#         )
#         # generate negative mask
#         # class_mask = torch.sum(prob_indices[:, :, :, :low_rank].eq(i), dim=3).bool() # torch.Size([16, 128, 128])
#         # negative_mask = class_mask * (label_all[:, i] == 0).bool() # torch.Size([16, 128, 128])
#         # negative_mask = negative_mask * ((mask_all != 0).bool()).bool() # torch.Size([16, 128, 128])

#         negative_mask = (label_all[:, i] == 0).bool() # torch.Size([16, 128, 128])
#         negative_mask = negative_mask * ((mask_all != 0).bool()).bool() # torch.Size([16, 128, 128])

#         keys = rep_all[negative_mask]
#         '''
#         i=0,keys.size() = torch.Size([18392, 256])
#         i=1,keys.size() = torch.Size([4273, 256])
#         i=2,keys.size() = torch.Size([13644, 256])
#         i=3,keys.size() = torch.Size([26809, 256])
#         i=4,keys.size() = torch.Size([10115, 256])
#         i=5,keys.size() = torch.Size([11119, 256])
#         '''

#         new_keys.append(
#             dequeue_and_enqueue(
#                 keys=keys,
#                 queue=memobank[i],
#                 queue_ptr=queue_ptrlis[i],
#                 queue_size=queue_size[i],
#             )
#         )

#         if valid_pixel_seg.sum() > 0:
#             seg_num_list.append(int(valid_pixel_seg.sum().item())) # [668, 18654, 10333, 1857, 15854]
#             valid_classes.append(i) # [0, 1, 2, 3, 4]
# #------------------------------------------------------------------------------


#     reco_loss = torch.tensor(0.0).cuda()
#     seg_proto = torch.cat(seg_proto_list)  # shape: [valid_classes, 256]
#     valid_seg = len(seg_num_list)  # number of valid classes

#     prototype = torch.zeros(
#         (prob_indices.shape[-1], num_queries, 1, num_feat)
#     ).cuda() # torch.Size([6, 256, 1, 256])

#     for i in range(valid_seg):
#         if (
#             #len(seg_feat_anchor_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0
#             memobank[valid_classes[i]][0].shape[0] > 0
#         ):
#             # select anchor pixel
#             # seg_feat_anchor_idx = torch.randint(
#             #     len(seg_feat_anchor_list[i]), size=(num_queries,)
#             # )
#             # anchor_feat = (
#             #     seg_feat_anchor_list[i][seg_feat_anchor_idx].clone().cuda()
#             # )
#             seg_feat_anchor_idx = torch.randint(
#                 len(seg_feat_all_list[valid_classes[i]]), size=(num_queries,)
#             )
#             anchor_feat = (
#                 seg_feat_all_list[valid_classes[i]][seg_feat_anchor_idx].clone().cuda()
#             )
#         else:
#             # in some rare cases, all queries in the current query class are easy
#             reco_loss = reco_loss + 0 * rep_all.sum()
#             continue

#         # apply negative key sampling from memory bank (with no gradients)

#         negative_feat = memobank[valid_classes[i]][0].clone().cuda()

#         negative_feat_idx = torch.randint(
#             len(negative_feat), size=(num_queries * num_negatives,)
#         )
#         negative_feat = negative_feat[negative_feat_idx]
#         negative_feat = negative_feat.reshape(
#             num_queries, num_negatives, num_feat
#         )
#         positive_feat = (
#             seg_proto[valid_classes[i]]
#             .unsqueeze(0)
#             .unsqueeze(0)
#             .repeat(num_queries, 1, 1)
#             .cuda()
#         )  # (num_queries, 1, num_feat)

#         if momentum_prototype is not None:
#             # if not (momentum_prototype == 0).all():
#             #     ema_decay = min(1 - 1 / i_iter, 0.999)
#             #     positive_feat = (
#             #         1 - ema_decay
#             #     ) * positive_feat + ema_decay * momentum_prototype[
#             #         valid_classes[i]
#             #     ]
#             prototype[valid_classes[i]] = positive_feat.clone()

#         all_feat = torch.cat(
#             (positive_feat, negative_feat), dim=1
#         )  # (num_queries, 1 + num_negative, num_feat)

#         seg_logits = torch.cosine_similarity(
#             anchor_feat.unsqueeze(1), all_feat, dim=2
#         )

#         reco_loss = reco_loss + F.cross_entropy(
#             seg_logits / temp, torch.zeros(num_queries).long().cuda()
#         )

#     return prototype, reco_loss / valid_seg

