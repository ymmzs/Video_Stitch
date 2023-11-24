import cv2
import numpy as np
import os
import json
# from get_matches import *
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist
import torch

device_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

# 视频帧resize
def resize_image(image):
    # 获取原图像的尺寸
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = image_pil.size

    # 指定裁剪的尺寸
    new_width = 1200
    new_height = 1200

    # 计算裁剪的坐标
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # 进行裁剪
    img_cropped = image_pil.crop((left, top, right, bottom))
    img_cropped = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)
    return img_cropped

# 将图像放置到新的画布，并计算更新后的坐标信息和变换矩阵
def warp_and_transform(full_canvas, image2, pts_src, pts_dst):
    # 将两幅图像放置在大画布中间
    height, width, _ = image2.shape
    start_y = (height * 3 - image2.shape[0]) // 2
    end_y = start_y + image2.shape[0]
    start_x = (width * 3 - image2.shape[1]) // 2
    end_x = start_x + image2.shape[1]
    full_canvas[start_y:end_y, start_x:end_x] = image2
    pts_src1 = pts_src + np.array([width * 2 // 2, height * 2 // 2])
    pts_dst1 = pts_dst + np.array([width * 2 // 2, height * 2 // 2])

    Homo, _ = cv2.findHomography(pts_dst1, pts_src1)
    warped_image = cv2.warpPerspective(full_canvas, Homo, (full_canvas.shape[1], full_canvas.shape[0]))

    return warped_image, Homo


def img_to_tensor(img1, img2, status):
    if status == 'opencv':
        transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([768, 768]),
        ])
    img1_tensor = (transform(img1)).unsqueeze(0)
    img2_tensor = (transform(img2)).unsqueeze(0)
    img_input_trans = torch.cat((img1_tensor, img2_tensor), 1)
    img_input_trans = img_input_trans.to(device_list[1])
    return img_input_trans


# 获取匹配点对
def find_matches(keypoints, homographies):
    keypoint1, keypoint2 = keypoints
    # 筛选真实匹配点
    keypoint1_np = keypoint1.cpu().detach().numpy()
    keypoint2_np = keypoint2.cpu().detach().numpy()

    # 单个单应性矩阵
    homographies_np = homographies.cpu().detach().numpy()
    kp1_projected = cv2.perspectiveTransform(keypoint1_np.reshape((1, -1, 2)), homographies_np)[0, :, :]
    dists = cdist(kp1_projected, keypoint2_np)
    min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)
    min1v = np.min(dists, axis=1)
    min1f = min2[min1v < 3]
    xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
    matches = np.intersect1d(min1f, xx)

    missing1 = np.setdiff1d(np.arange(keypoint1_np.shape[0]), min1[matches])
    missing2 = np.setdiff1d(np.arange(keypoint2_np.shape[0]), matches)

    MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
    MN2 = np.concatenate([missing1[np.newaxis, :], (len(keypoint2)) * np.ones((1, len(missing1)), dtype=np.int64)])
    MN3 = np.concatenate([(len(keypoint1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
    # 获取匹配点对和未匹配点对
    all_matches = MN.transpose(1, 0)
    all_matches = torch.from_numpy(all_matches)

    all_unmatches = np.concatenate([MN2, MN3], axis=1)
    all_unmatches = all_unmatches.transpose(1, 0)
    all_unmatches = torch.from_numpy(all_unmatches)

    return all_matches, all_unmatches


# def get_folder_mosaic(folder_path, gt_path, early_stop_path):
#     image_files = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg'))])
#     filename_1 = image_files[0]
#     image1_path = os.path.join(folder_path, filename_1)
#     image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
#     filename_1_json = filename_1.replace('.jpg', '.json')
#     image1_json_path = os.path.join(gt_path, filename_1_json)
#     with open(image1_json_path, 'r') as file1:
#         image1_json = json.load(file1)
#
#     H_m = None
#     height, width, _ = image1.shape
#     result = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
#     start_y = (height * 3 - image1.shape[0]) // 2
#     end_y = start_y + image1.shape[0]
#     start_x = (width * 3 - image1.shape[1]) // 2
#     end_x = start_x + image1.shape[1]
#     result[start_y:end_y, start_x:end_x] = image1
#     # result[height//2:height//2+height, width//2:width//2+width] = image1
#
#     for i in range(1, len(image_files)):
#         filename_2 = image_files[i]
#         image2_path = os.path.join(folder_path, filename_2)
#         image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
#         filename_2_json = filename_2.replace('.jpg', '.json')
#         image2_json_path = os.path.join(gt_path, filename_2_json)
#         with open(image2_json_path, 'r') as file2:
#             image2_json = json.load(file2)
#
#         pts_src1, pts_dst1 = get_homography1(image1_json, image2_json)
#         # 创建一个足够大的画布来容纳两个图像的完整区域
#         full_canvas = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
#         warped_image2, H_m = warp_and_transform(full_canvas, image2, pts_src1, pts_dst1)
#
#         if filename_2 == early_stop_path:
#             return result, H_m
#         result = np.maximum(result, warped_image2)
#
#     cv2.imwrite(os.path.join(folder_path, "result_image.jpg"), result)
#     return result, H_m


