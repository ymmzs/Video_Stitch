import torch
# from memory_profiler import profile
import time

def pairwise_distance(x1, x2, p=2, eps=1e-6):
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."

    return 1 - torch.cosine_similarity(x1, x2, dim=1)


def triplet_margin_loss_gor(anchor, positive, negative1, negative2, margin=1.0, p=2, eps=1e-6, swap=False):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative1.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative2.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'

    dis_a_p = pairwise_distance(anchor, positive, p, eps)  # D(da, dp)
    dis_p_a = pairwise_distance(positive, anchor, p, eps)  # D(dp, da)
    dis_a_u = pairwise_distance(anchor, negative1, p, eps)  # D(da, du)
    dis_p_v = pairwise_distance(positive, negative2, p, eps)  # D(dp, dv)

    loss_item1 = torch.clamp(margin + dis_a_p - dis_a_u, min=0)
    loss_item2 = torch.clamp(margin + dis_p_a - dis_p_v, min=0)

    loss = loss_item1 + loss_item2

    return loss


def descriptor_loss(keypoints1, keypoints2, descriptor1, descriptor2, matches):
    keypoints1 = keypoints1.squeeze(0)
    keypoints2 = keypoints2.squeeze(0)
    descriptor1 = descriptor1.squeeze(0)
    descriptor2 = descriptor2.squeeze(0)
    matches = matches.squeeze(0)
    # 计算特征点之间的欧氏距离
    kpts0_kpts1_D = torch.cdist(keypoints1, keypoints2, p=2)

    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    descriptor1 = descriptor1.transpose(1, 0)
    descriptor2 = descriptor2.transpose(1, 0)

    # 获取正样本对的特征点坐标和描述符
    da = descriptor1[idx1]   # [Num, Dim]
    dp = descriptor2[idx2]

    # 计算du和dv，即anchor在另一幅图像中的最近的非匹配描述符
    # 在anchor所在图像中找到最近的非匹配描述符
    min_indices_nonmatching_1 = torch.argsort(kpts0_kpts1_D[idx1], dim=1)
    du = descriptor2[min_indices_nonmatching_1[:, 1]]

    # 在positive所在图像中找到最近的非匹配描述符
    min_indices_nonmatching_2 = torch.argsort(kpts0_kpts1_D[:, idx2], dim=0)
    dv = descriptor1[min_indices_nonmatching_2[1]]

    loss_descriptor = triplet_margin_loss_gor(da, dp, du, dv, margin=1)
    return loss_descriptor.sum()/len(loss_descriptor)


# def matches_loss(scores, all_matches, all_unmatches):
#     loss_match = []
#     loss_unmatch = []
#
#     for i in range(len(all_matches)):
#         for j in range(len(all_matches[i])):
#             x = all_matches[i][j][0]
#             y = all_matches[i][j][1]
#             loss_match.append(-scores[i, x, y])
#
#     for i in range(len(all_unmatches)):
#         for j in range(len(all_unmatches[i])):
#             m = all_unmatches[i][j][0]
#             n = all_unmatches[i][j][1]
#             loss_unmatch.append(-scores[i, m, n])
#
#     if len(loss_match) == 0 or len(loss_unmatch) == 0:  # no matches on retina.
#         print("There is no matching pair on the vessel.")
#         return None
#
#     else:
#         all_match_loss = torch.mean(torch.stack(loss_match))
#         all_unmatch_loss = torch.mean(torch.stack((loss_unmatch)))
#         loss_all_mean = all_match_loss + all_unmatch_loss
#
#         return loss_all_mean

def matches_loss(scores, all_matches, all_unmatches):
    loss_match = []
    loss_unmatch = []
    epsilon = 1e-6
    for i in range(len(all_matches)):
        for j in range(len(all_matches[i])):
            x = all_matches[i][j][0]
            y = all_matches[i][j][1]
            loss_match.append(-torch.log(scores[i][x][y].exp() + epsilon))

    for i in range(len(all_unmatches)):
        for j in range(len(all_unmatches[i])):
            m = all_unmatches[i][j][0]
            n = all_unmatches[i][j][1]
            loss_unmatch.append(-torch.log(scores[i][m][n].exp() + epsilon))
    if len(loss_match) == 0 or len(loss_unmatch) == 0:  # no matches on retina.
        print("There is no matching pair on the vessel.")
        return None

    else:
        all_match_loss = torch.mean(torch.stack(loss_match))
        all_unmatch_loss = torch.mean(torch.stack((loss_unmatch)))
        loss_all_mean = 0.45 * all_match_loss + 1.0 * all_unmatch_loss

        return loss_all_mean


def keypoints_loss(keypoints1, keypoints1_proj, matches):
    keypoints1 = keypoints1.squeeze(0)
    keypoints1_proj = keypoints1_proj.squeeze(0)
    matches = matches.squeeze(0)
    idxs1 = matches[:, 0]

    kp1_projected_tensor = keypoints1_proj.clone().detach()
    kpts_loss = pairwise_distance(keypoints1[idxs1], kp1_projected_tensor[idxs1])
    kpts_loss1 = (1/len(kpts_loss)*len(kpts_loss))*kpts_loss.sum()

    return kpts_loss1


# if __name__ == '__main__':
#     # 特征提取网络
#     model_feature = SuperRetina()
#     model_save_path = '/workspace/data/ResNet_Attention_Minidata/save_model/SuperRetina.pth'
#     checkpoint = torch.load(model_save_path, map_location=device_list[0])
#     model_feature.load_state_dict(checkpoint['net'])
#     model_feature.to(device_list[0])
#
#     refer_image_orig = cv2.imread("/workspace/data/ResNet_Attention_FIRE/DATASET/TRAIN/S70_1.jpg", cv2.IMREAD_COLOR)
#     query_image_orig = cv2.imread("/workspace/data/ResNet_Attention_FIRE/DATASET/TRAIN/S70_2.jpg", cv2.IMREAD_COLOR)
#     height_orig = refer_image_orig.shape[0]
#
#     refer_img_gray = cv2.cvtColor(refer_image_orig, cv2.COLOR_BGR2GRAY)
#     query_img_gray = cv2.cvtColor(query_image_orig, cv2.COLOR_BGR2GRAY)
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize([512, 512]),
#     ])
#
#     refer_image = (transform(refer_img_gray)).unsqueeze(0)
#     query_image = (transform(query_img_gray)).unsqueeze(0)
#
#     img_input_trans = torch.cat((refer_image, query_image))  # [2, 1, 512, 512]
#     img_input_trans = img_input_trans.to(device_list[0])
#
#     data_pred = get_real_kpts(model_feature, img_input_trans)
#
#     keypoint1 = data_pred['keypoints'][0]
#     keypoint2 = data_pred['keypoints'][1]
#     scores1 = data_pred['scores'][0]  # 特征点的置信度
#     scores2 = data_pred['scores'][1]
#     descriptor1 = data_pred['descriptors'][0]  # 特征点的描述子
#     descriptor2 = data_pred['descriptors'][1]
#
#     # 图像的变换矩阵
#     point_path = os.path.join("/workspace/data/ResNet_Attention_FIRE/Ground Truth/control_points_S70_1_2.txt")
#     scale_factor = 512 / height_orig
#     homo = get_homography1(point_path, scale_factor)
#
#     keypoint1_np = np.array(keypoint1).astype(np.float32)  # cv2.perspectiveTransform的输入为浮点数
#     keypoint2_np = np.array(keypoint2).astype(np.float32)
#
#     kp1_projected = cv2.perspectiveTransform(keypoint1_np.reshape((1, -1, 2)), homo)[0, :, :]
#     dists = cdist(kp1_projected, keypoint2_np)
#     min1 = np.argmin(dists, axis=0)
#     min2 = np.argmin(dists, axis=1)
#     min1v = np.min(dists, axis=1)
#     min1f = min2[min1v < 3]
#     xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
#     matches = np.intersect1d(min1f, xx)
#
#     missing1 = np.setdiff1d(np.arange(keypoint1_np.shape[0]), min1[matches])
#     missing2 = np.setdiff1d(np.arange(keypoint2_np.shape[0]), matches)
#
#     MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
#     MN2 = np.concatenate([missing1[np.newaxis, :], (len(keypoint2)) * np.ones((1, len(missing1)), dtype=np.int64)])
#     MN3 = np.concatenate([(len(keypoint1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
#     # 获取匹配点对和未匹配点对
#     all_matches = MN.transpose(1, 0)
#     all_unmatches = np.concatenate([MN2, MN3], axis=1)
#     all_unmatches = all_unmatches.transpose(1, 0)
#
#     loss = descriptor_loss(keypoint1, keypoint2, descriptor1, descriptor2, all_matches)
#
#     # kpts loss.
#     idxs1 = all_matches[:, 0]
#     idxs2 = all_matches[:, 1]
#     kp1_projected_tensor = torch.tensor(kp1_projected)
#     kpts_loss = pairwise_distance(keypoint1[idxs1], kp1_projected_tensor[idxs1])
#     kpts_loss1 = (30/len(kpts_loss)*len(kpts_loss))*kpts_loss.sum()
