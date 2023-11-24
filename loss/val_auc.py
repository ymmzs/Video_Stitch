import numpy as np
import cv2
from utils import device_list
from torchvision import transforms
import torch
import time
import os
import yaml

config_path = 'model/ROPconfig.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

data_config = config['dataset_params']

image_height = data_config['image_height']
image_width = data_config['image_width']

def img_transforms(img):
    img_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([image_height, image_width]),
    ])
    img_tensor = img_trans(img)
    img_tensor = img_tensor.unsqueeze(0)  # 将图像构造成（batch, channel, height, width） --> (1, 3, 512, 512)
    return img_tensor


def compute_homo(kpt1, kpt0, matches1, matches0):
    homography_matrix = None
    inliers_num_rate = 0

    matches0 = matches0.detach().cpu().numpy()
    matches1 = matches1.detach().cpu().numpy()
    match_point0 = []
    match_point1 = []
    for i in range(matches1.shape[0]):
        if matches1[i] != -1 and matches0[matches1[i]] == i:
            cv_kpts0 = torch.tensor(
                (int(kpt1[i][0] / data_config['image_width'] * data_config['image_orig_width']),
                 int(kpt1[i][1] / data_config['image_width'] * data_config['image_orig_width'])))
            cv_kpts1 = torch.tensor((int(kpt0[matches1[i]][0] / data_config['image_width'] * data_config['image_orig_width']),
                                     int(kpt0[matches1[i]][1] / data_config['image_width'] * data_config['image_orig_width'])))

            match_point0.append(cv_kpts0)
            match_point1.append(cv_kpts1)

    if len(match_point0) >= 4:
        match_p0 = torch.stack(match_point0)
        match_kpts0 = match_p0.cpu().numpy()
        match_p1 = torch.stack(match_point1)
        match_kpts1 = match_p1.cpu().numpy()

        homography_matrix, mask = cv2.findHomography(match_kpts0, match_kpts1, cv2.LMEDS)
        inliers_num_rate = mask.sum() / len(mask.ravel())

    return homography_matrix, inliers_num_rate


def valiation_data(data_dict, matches0, matches1, img_name):
    big_num = 1e6
    avg_dist = []

    keypoint0, keypoint1 = data_dict
    H_m, inliers_num_rate = compute_homo(keypoint1, keypoint0, matches1, matches0)

    if H_m is None or inliers_num_rate < 1e-6:
        avg_dist.append(big_num)
    else:
        # 自己构造的ground truth
        # all_match11 = data_dict['all_matches'][i]
        # all_match11_np = all_match11.cpu().detach().numpy()
        # matches = [cv2.DMatch(m[0], m[1], 0) for m in all_match11_np]
        # cv_kpts0 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in keypoint0[i]]
        # cv_kpts1 = [cv2.KeyPoint(int(i[0]), int(i[1]), 30) for i in keypoint1[i]]
        # match_points = np.array([(cv_kpts0[m.queryIdx].pt, cv_kpts1[m.trainIdx].pt) for m in matches])
        # raw = match_points[:, 1, :]
        # dst = match_points[:, 0, :]

        # FIRE数据集中的ground truth
        point_path = os.path.join(data_config['homo_matrix_path'], "control_points_{}_1_2.txt".format(img_name[0]))
        points_gd = np.loadtxt(point_path)
        raw = np.zeros([len(points_gd), 2])
        dst = np.zeros([len(points_gd), 2])
        raw[:, 0] = points_gd[:, 2]
        raw[:, 1] = points_gd[:, 3]
        dst[:, 0] = points_gd[:, 0]
        dst[:, 1] = points_gd[:, 1]

        dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_m)
        dst_pred = dst_pred.squeeze()

        dis = (dst - dst_pred) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        avg_dist.append(dis.mean())

    avg_dist = np.array(avg_dist)

    return avg_dist


def compute_auc(s_error):

    limit = 25
    gs_error = np.zeros(limit + 1)

    accum_s = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        accum_s = accum_s + gs_error[i]

    auc_s = accum_s / (limit * 100)

    return auc_s
