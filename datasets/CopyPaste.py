import cv2
import numpy as np
from style_transfer import style_transfer
# import os
# # import random
# import string

def Large_Scale_Jittering(im1, im2, mask1, mask2, mask_bin, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = im1.shape
 
    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    im1 = cv2.resize(im1, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    im2 = cv2.resize(im2, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask1 = cv2.resize(mask1, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
 
    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        im1_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        im2_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask1_pad = np.zeros((h, w), dtype=np.uint8)
        mask2_pad = np.zeros((h, w), dtype=np.uint8)
        mask_bin_pad = np.zeros((h, w), dtype=np.uint8)
        im1_pad[y:y+h_new, x:x+w_new, :] = im1
        im2_pad[y:y+h_new, x:x+w_new, :] = im2
        mask1_pad[y:y+h_new, x:x+w_new] = mask1
        mask2_pad[y:y+h_new, x:x+w_new] = mask2
        mask_bin_pad[y:y+h_new, x:x+w_new] = mask_bin
        return im1_pad, im2_pad, mask1_pad, mask2_pad, mask_bin_pad
    else:  # crop
        im1_crop = im1[y:y+h, x:x+w, :]
        im2_crop = im2[y:y+h, x:x+w, :]
        mask1_crop = mask1[y:y+h, x:x+w]
        mask2_crop = mask2[y:y+h, x:x+w]
        mask_bin_crop = mask_bin[y:y+h, x:x+w]
        return im1_crop, im2_crop, mask1_crop, mask2_crop, mask_bin_crop

def avoid_overwrite(copy_mask, target_mask, kernel):
    
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    overwrite_contours = []
    # print(len(copy_contours))
    for copy_contour in copy_contours:
        
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        
        copy_contour_mask = cv2.dilate(copy_contour_mask, kernel, iterations = 1)
        
        copy_index = np.where(copy_contour_mask==255)
        copy_index_XmergeY = copy_index[0]*1.0+copy_index[1]*0.001
        
        for target_contour in target_contours:
            
            target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
            cv2.fillPoly(target_contour_mask, [target_contour], (255))
            target_contour_mask = cv2.dilate(target_contour_mask, kernel, iterations = 1)
            
            target_index = np.where(target_contour_mask==255)
            target_index_XmergeY = target_index[0]*1.0+target_index[1]*0.001
            
            # 若contour1和contour2相交
            if True in np.isin(copy_index_XmergeY,target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                # print(1)
                break
                
    cv2.fillPoly(copy_mask, overwrite_contours, (0))
    # cv2.imwrite("22222.png",copy_mask)
    return copy_mask

def img_add(img_src, img_main, mask_src, isdilate=False):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
        
    sub_img_src = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask_src)
    sub_img_src_wh = cv2.resize(sub_img_src, (w, h),interpolation=cv2.INTER_NEAREST)
    mask_src_wh = cv2.resize(mask_src, (w, h), interpolation=cv2.INTER_NEAREST)
    sub_img_main = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8), mask=mask_src_wh)
    img_main = img_main - sub_img_main + sub_img_src_wh
    return img_main


def copy_paste(img_copy, mask_copy, mask_bin_copy, img_target, mask_target, mask_bin_target, random):
    
    img_copy = style_transfer(img_copy, img_target)
    if random<0.25 :
        # 水平翻转
        img_copy = cv2.flip(img_copy, 1)
        mask_copy = cv2.flip(mask_copy, 1)
        mask_bin_copy = cv2.flip(mask_bin_copy, 1)
    elif (random>=0.25) and (random<0.5) :
        # 垂直翻转
        img_copy = cv2.flip(img_copy, 0)
        mask_copy = cv2.flip(mask_copy, 0)
        mask_bin_copy = cv2.flip(mask_bin_copy, 0)
    elif (random>=0.5) and (random<0.75) :
        # 对角翻转
        img_copy = cv2.flip(img_copy, -1)
        mask_copy = cv2.flip(mask_copy, -1)
        mask_bin_copy = cv2.flip(mask_bin_copy, -1)
    else:
        img_copy=np.rot90(img_copy).copy()
        mask_copy=np.rot90(mask_copy).copy()
        mask_bin_copy=np.rot90(mask_bin_copy).copy()
    
    # print(mask.shape, mask_filp.shape)
    # 膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    # 若和目标有重叠,则不进行粘贴
    mask_bin_copy = avoid_overwrite(mask_bin_copy, mask_bin_target, kernel)
    # cv2.imwrite("11111.png",mask_filp)
    mask_bin_copy_dilate = cv2.dilate(mask_bin_copy, kernel, iterations = 1)
    
    img = img_add(img_copy, img_target, mask_bin_copy_dilate, isdilate=True)
    mask = img_add(mask_copy, mask_target, mask_bin_copy, isdilate=False)
    mask_bin = img_add(mask_bin_copy, mask_bin_target, mask_bin_copy, isdilate=False)
    
    return img, mask, mask_bin

def copy_paste_self(img, mask, random):
    
    
    if random<0.25 :
        # 水平翻转
        img_filp = cv2.flip(img, 1)
        mask_filp = cv2.flip(mask, 1)
    elif (random>=0.25) and (random<0.5) :
        # 垂直翻转
        img_filp = cv2.flip(img, 0)
        mask_filp = cv2.flip(mask, 0)
    elif (random>=0.5) and (random<0.75) :
        # 对角翻转
        img_filp = cv2.flip(img, -1)
        mask_filp = cv2.flip(mask, -1)
    else:
        img_filp=np.rot90(img).copy()
        mask_filp=np.rot90(mask).copy()
    
    # print(mask.shape, mask_filp.shape)
    # 膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    # 若和目标有重叠,则不进行粘贴
    mask_filp = avoid_overwrite(mask_filp, mask, kernel)
    # cv2.imwrite("11111.png",mask_filp)
    mask_filp_dilate = cv2.dilate(mask_filp, kernel, iterations = 1)
    
    img = img_add(img_filp, img, mask_filp_dilate, isdilate=True)
    mask = img_add(mask_filp, mask, mask_filp, isdilate=False)
    
    return mask, img

# # 复制粘贴的图像和标签
# mask_src = cv2.imread(r"41_1_label.png",0)
# img_src = cv2.imread(r"41_1.png")

# # 粘贴背景图像和标签
# mask_main = cv2.imread(r"23_2_label.png",0)
# img_main = cv2.imread(r"23_2.png")

# random = np.random.random()
# # 复制粘贴数据增强
# mask, img = copy_paste(img_src, mask_src, img_main, mask_main, random)

# cv2.imwrite("copy_paste_label.png", mask)
# cv2.imwrite("copy_paste.png",img)




# mask = cv2.imread(r"41_1_label.png",0)
# # mask[mask==255]=1
# img = cv2.imread(r"41_1.png")

# mask, img = copy_paste_self(img, mask)
# cv2.imwrite("41_1_label_.png", mask)
# cv2.imwrite("41_1_.png",img)