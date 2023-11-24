import numpy as np
import json
from torchvision import transforms
from utils import *
from model.SuperRetina.SuperRetina import get_real_kpts_imgs
"""
1. 手动标注的数据计算的匹配点
"""
def get_homography_manual(image1_json_path, image2_json_path):
    with open(image1_json_path, 'r') as file1:
        data1 = json.load(file1)
    with open(image2_json_path, 'r') as file2:
        data2 = json.load(file2)

    # 创建一个字典，用于存储相同标签的坐标信息
    label_to_coordinates = {}
    # 初始化点对数量为0
    point_pair_count = 0
    # 遍历json_file1.json文件中的标签和坐标信息
    for item1 in data1['shapes']:
        label1 = item1['label']
        coordinates1 = item1['points'][0]
        # 遍历json_file2.json文件中的标签和坐标信息
        for item2 in data2['shapes']:
            label2 = item2['label']
            coordinates2 = item2['points'][0]
            # 如果标签相同，则保存坐标信息
            if label1 == label2:
                point_pair_count += 1
                key = str(label1)  # 使用标签作为字典的键
                x1, y1 = map(lambda x: round(x), coordinates1)  # 取整
                x2, y2 = map(lambda x: round(x), coordinates2)
                # 将坐标信息取整并格式化为带有小数点三位的字符串
                formatted_coordinates = f"{int(x1):03d}.{int(x1 % 1 * 1000):03d} {int(y1):03d}.{int(y1 % 1 * 1000):03d} " \
                                        f"{int(x2):03d}.{int(x2 % 1 * 1000):03d} {int(y2):03d}.{int(y2 % 1 * 1000):03d}"
                if key in label_to_coordinates:
                    label_to_coordinates[key].append(formatted_coordinates)
                else:
                    label_to_coordinates[key] = [formatted_coordinates]

    matches = []
    if point_pair_count >= 8:
        for label, coordinates_list in label_to_coordinates.items():
            for coordinates in coordinates_list:
                x1, y1, x2, y2 = map(float, coordinates.split())
                matches.append((x1, y1, x2, y2))
    else:
        matches.append(([], [], [], []))

    pts_src = np.array([(m[0], m[1]) for m in matches], dtype=np.float32)
    pts_dst = np.array([(m[2], m[3]) for m in matches], dtype=np.float32)
    # H, _ = cv2.findHomography(pts_dst, pts_src)       # 切换到更大的坐标空间

    return pts_src, pts_dst

"""
2. opencv计算的匹配点
"""


def get_homography_opencv(feat_config, model_feat, data1, data2):
    data1 = data1[:, :, 1]
    data2 = data2[:, :, 1]
    data_tensor = img_to_tensor(data1, data2, 'opencv')
    data_pred = get_real_kpts_imgs(feat_config, model_feat, data_tensor)

    kpt0, kpt1 = data_pred['keypoints'][0]
    desc0, desc1 = data_pred['descriptors'][0]
    desc0, desc1 = desc0.transpose(1, 0), desc1.transpose(1, 0)
    desc0, desc1 = desc0.detach().numpy(), desc1.detach().numpy()
    # mapping keypoints to scaled keypoints
    cv_kpts_refer = [cv2.KeyPoint(int(i[0]), int(i[1]), 30)
                     for i in kpt0]
    cv_kpts_query = [cv2.KeyPoint(int(i[0]), int(i[1]), 30)
                     for i in kpt1]

    goodMatch = []
    status = []
    matches = []
    knn_matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_thresh = 0.8
    matches = knn_matcher.knnMatch(desc0, desc1, k=2)
    for m, n in matches:
        if m.distance < knn_thresh * n.distance:
            goodMatch.append(m)

    if len(goodMatch) >= 4:
        src_pts = [cv_kpts_refer[m.queryIdx].pt for m in goodMatch]
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = [cv_kpts_query[m.trainIdx].pt for m in goodMatch]
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        # H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
    else:
        src_pts = np.zeros((0, 1, 2), dtype=np.float32)
        dst_pts = np.zeros((0, 1, 2), dtype=np.float32)

    return src_pts, dst_pts

"""
3. superglue计算的匹配点
"""
def get_homography_superglue(feat_config, model_feat, match_model, data1, data2):
    data1 = data1[:, :, 1]
    data2 = data2[:, :, 1]
    data_tensor = img_to_tensor(data1, data2, 'superglue')
    data = get_real_kpts_imgs(feat_config, model_feat, data_tensor)
    for key in data:
        data[key] = [(item[0].to(device_list[1]), item[1].to(device_list[1])) for item in data[key]]

    superglue_input = {
        'data_pred': data,
        'homography': None,
        'images': data_tensor,
        'image_prefix_name': None
    }
    match_res = match_model(superglue_input, is_training=False)
    matches0 = match_res[0]['matches0'].detach().cpu().numpy()
    matches1 = match_res[0]['matches1'].detach().cpu().numpy()

    match_point0_re = []
    match_point1_re = []
    kpt0, kpt1 = data['keypoints'][0]
    model_height, model_width = feat_config['image_height'], feat_config['image_width']
    image_height_orig, image_width_orig = feat_config['image_orig_height'], feat_config['image_orig_width']
    for i in range(matches0.shape[0]):
        if matches0[i] != -1 and matches1[matches0[i]] == i:
            # 记录原图（H*W）相当于模型（512*512）的坐标
            cv_kpts0 = torch.tensor(
                (int(kpt0[i][0] / model_width * image_width_orig), int(kpt0[i][1] / model_height * image_height_orig)))
            cv_kpts1 = torch.tensor((int(kpt1[matches0[i]][0] / model_width * image_width_orig),
                                     int(kpt1[matches0[i]][1] / model_height * image_height_orig)))

            match_point0_re.append(cv_kpts0)
            match_point1_re.append(cv_kpts1)

    if len(match_point0_re) >= 50:        # optional
        match_p0 = torch.stack(match_point0_re)
        src_pts = match_p0.cpu().numpy()
        match_p1 = torch.stack(match_point1_re)
        dst_pts = match_p1.cpu().numpy()

        # H_m, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
    else:
        src_pts = np.zeros((0, 1, 2), dtype=np.float32)
        dst_pts = np.zeros((0, 1, 2), dtype=np.float32)

    return src_pts, dst_pts