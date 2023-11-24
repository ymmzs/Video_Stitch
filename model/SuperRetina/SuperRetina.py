import random

from model.SuperRetina.pke_module import pke_learn

import torch.nn as nn

from model.SuperRetina.dice_loss import DiceLoss
from model.SuperRetina.triplet_loss import triplet_margin_loss_gor

from model.SuperRetina.common_util import remove_borders, simple_nms, sample_descriptors
from model.SuperRetina.train_util import get_gaussian_kernel, affine_images

import torch
from torch.nn import functional as F

from model.SuperRetina.common_util import sample_keypoint_desc, nms
from model.SuperRetina.record_module import update_value_map



def mapping_points(grid, points, h, w):
    """ Using grid_inverse to apply affine transform on geo_points
        :return point set and its corresponding affine point set
    """

    grid_points = [(grid[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                   enumerate(points)]
    filter_points = []
    affine_points = []
    for s, k in enumerate(grid_points):  # filter bad geo_points
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        gp = grid_points[s][idx]
        gp[:, 0] = (gp[:, 0] + 1) / 2 * (w - 1)
        gp[:, 1] = (gp[:, 1] + 1) / 2 * (h - 1)
        affine_points.append(gp)
        filter_points.append(points[s][idx])

    return filter_points, affine_points


def content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                   affine_geo_points, content_thresh=0.7, scale=8):
    """
    content-based matching in paper
    :param descriptor_pred: descriptors of input_image images
    :param affine_descriptor_pred: descriptors of affine images
    :param geo_points:
    :param affine_geo_points:
    :param content_thresh:
    :param scale: down sampling size of descriptor_pred
    :return: content-filtered keypoints
    """

    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                   for k, d in zip(geo_points, descriptor_pred)]
    aff_descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                       for k, d in zip(affine_geo_points, affine_descriptor_pred)]
    content_points = []
    affine_content_points = []
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    for i in range(len(dist)):
        D = dist[i]
        if len(D) <= 1:
            content_points.append([])
            affine_content_points.append([])
            continue
        val, ind = torch.topk(D, 2, dim=1, largest=False)

        arange = torch.arange(len(D))
        # rule1 spatial correspondence
        c1 = ind[:, 0] == arange.to(ind.device)
        # rule2 pass the ratio evalute
        c2 = val[:, 0] < val[:, 1] * content_thresh

        check = c2 * c1
        content_points.append(geo_points[i][check])
        affine_content_points.append(affine_geo_points[i][check])
    return content_points, affine_content_points


def geometric_filter(affine_detector_pred, points, affine_points, max_num=1024, geometric_thresh=0.5):
    """
    geometric matching in paper
    :param affine_detector_pred: geo_points probability of affine image
    :param points: nms results of input_image image
    :param affine_points: nms results of affine image
    :param max_num: maximum number of learned keypoints
    :param geometric_thresh:
    :return: geometric-filtered keypoints
    """
    geo_points = []
    affine_geo_points = []
    for s, k in enumerate(affine_points):
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        check = sample_aff_values.squeeze() >= geometric_thresh
        geo_points.append(points[s][check][:max_num])
        affine_geo_points.append(k[check][:max_num])

    return geo_points, affine_geo_points


def pke_learn(detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
              affine_descriptor_pred, kernel, loss_cal, label_point_positions,
              value_map, config, PKE_learn=True):
    """
    pke process used for detector
    :param detector_pred: probability map from raw image
    :param descriptor_pred: prediction of descriptor_pred network
    :param kernel: used for gaussian heatmaps
    :param mask_kernel: used for masking initial keypoints
    :param grid_inverse: used for inverse
    :param loss_cal: loss_super (default is dice)
    :param label_point_positions: positions of keypoints on labels
    :param value_map: value map for recoding and selecting learned geo_points
    :param pke_learn: whether to use PKE
    :return: loss_super of detector, num of additional geo_points, updated value maps and enhanced labels
    """
    # used for masking initial keypoints on enhanced labels
    initial_label = F.conv2d(label_point_positions, kernel,
                             stride=1, padding=(kernel.shape[-1] - 1) // 2)
    initial_label[initial_label > 1] = 1

    if not PKE_learn:
        return loss_cal(detector_pred, initial_label.to(detector_pred)), 0, None, None, initial_label

    nms_size = config['nms_size']
    nms_thresh = config['nms_thresh']
    scale = 8

    enhanced_label = None
    geometric_thresh = config['geometric_thresh']
    content_thresh = config['content_thresh']
    with torch.no_grad():
        h, w = detector_pred.shape[2:]

        # number of learned points
        number_pts = 0
        points = nms(detector_pred, nms_thresh=nms_thresh, nms_size=nms_size,
                     detector_label=initial_label, mask=True)

        # geometric matching
        points, affine_points = mapping_points(grid_inverse, points, h, w)
        geo_points, affine_geo_points = geometric_filter(affine_detector_pred, points, affine_points,
                                                         geometric_thresh=geometric_thresh)

        # content matching
        content_points, affine_contend_points = content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                                                               affine_geo_points, content_thresh=content_thresh,
                                                               scale=scale)
        enhanced_label_pts = []
        for step in range(len(content_points)):
            # used to combine initial points and learned points
            positions = torch.where(label_point_positions[step, 0] == 1)
            if len(positions) == 2:
                positions = torch.cat((positions[1].unsqueeze(-1), positions[0].unsqueeze(-1)), -1)
            else:
                positions = positions[0]

            final_points = update_value_map(value_map[step], content_points[step], config)

            # final_points = torch.cat((final_points, positions))

            temp_label = torch.zeros([h, w]).to(detector_pred.device)

            temp_label[final_points[:, 1], final_points[:, 0]] = 0.8
            temp_label[positions[:, 1], positions[:, 0]] = 1
            enhanced_kps = nms(temp_label.unsqueeze(0).unsqueeze(0), nms_thresh, nms_size)[0]
            number_pts += len(enhanced_kps) - len(positions)

            temp_label[:] = 0
            temp_label[enhanced_kps[:, 1], enhanced_kps[:, 0]] = 1

            enhanced_label_pts.append(temp_label.unsqueeze(0).unsqueeze(0))

            temp_label = F.conv2d(temp_label.unsqueeze(0).unsqueeze(0), kernel, stride=1,
                                  padding=(kernel.shape[-1] - 1) // 2)  # generating gaussian heatmaps
            temp_label[temp_label > 1] = 1

            if enhanced_label is None:
                enhanced_label = temp_label
            else:
                enhanced_label = torch.cat((enhanced_label, temp_label))

    enhanced_label_pts = torch.cat(enhanced_label_pts)
    affine_pred_inverse = F.grid_sample(affine_detector_pred, grid_inverse, align_corners=True)

    loss1 = loss_cal(detector_pred, enhanced_label)  # L_geo
    loss2 = loss_cal(detector_pred, affine_pred_inverse)  # L_clf
    # pred_mask = (enhanced_label > 0) & (affine_pred_inverse != 0)
    # loss2 = loss_cal(detector_pred[pred_mask], affine_pred_inverse[pred_mask])  # L_clf

    # mask_pred = grid_inverse
    # loss2 = loss_cal(detector_pred[mask_pred], affine_pred_inverse[mask_pred])  # L_clf

    loss = loss1+loss2

    return loss, number_pts, value_map, enhanced_label_pts, enhanced_label


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class SuperRetina(nn.Module):
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = True
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # Detector Head
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)

        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config

            self.nms_size = config['nms_size']
            self.nms_thresh = config['nms_thresh']
            self.scale = 8

            self.dice = DiceLoss()

            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)

    def network(self, x):
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        desc = self.trans_conv(desc)

        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc

    def forward(self, x):
        """
        In interface phase, only need to input x
        :param x: retinal images
        :param label_point_positions: positions of keypoints on labels
        :param value_map: value maps, used to record history learned geo_points
        :param learn_index: index of input data with detector labels
        :param phase: distinguish preprocess
        :return: if training, return loss_super, else return predictions
        """

        detector_pred, descriptor_pred = self.network(x)

        return detector_pred, descriptor_pred


def mySuperRetina(model, inputs, descs_seg, scores_seg):

    # 拼接后的结果
    detector_pred, descriptor_pred = model(inputs)

    scores = simple_nms(detector_pred, 10)

    b, _, h, w = detector_pred.shape
    scores = scores.reshape(-1, h, w)
    scores_seg = scores_seg.reshape(-1, h, w)

    keypoints = [
        torch.nonzero(s > 0.01)
        for s in scores]

    scores = [s1[tuple(k1.t())] for s1, k1 in zip(scores, keypoints)]
    scores_seg = [s2[tuple(k2.t())] for s2, k2 in zip(scores_seg, keypoints)]

    # Discard keypoints near the image borders
    keypoints1, scores = list(zip(*[
        remove_borders(k1, s1, 4, h, w)
        for k1, s1 in zip(keypoints, scores)]))

    _, scores_seg = list(zip(*[
        remove_borders(k2, s2, 4, h, w)
        for k2, s2 in zip(keypoints, scores_seg)]))

    keypoints1 = [torch.flip(k, [1]).float().data for k in keypoints1]
    descriptors = [sample_keypoint_desc(k1[None], d1[None], 8)[0].cpu()
                   for k1, d1 in zip(keypoints1, descriptor_pred)]

    descriptors_seg = [sample_keypoint_desc(k2[None], d2[None], 8)[0].cpu()
                   for k2, d2 in zip(keypoints1, descs_seg)]

    keypoints1 = [k.cpu() for k in keypoints1]

    return {
        'keypoints': keypoints1,
        'descriptors': descriptors,  # [维度，个数]
        'descriptors_seg': descriptors_seg,
        'scores': scores,
        'scores_seg': scores_seg,
    }


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def get_real_kpts(config, model, inputs):
    # 拼接后的结果
    detector_pred, descriptor_pred = model(inputs)
    scores = simple_nms(detector_pred, config['nms_radius'])
    b, _, h, w = detector_pred.shape
    scores = scores.reshape(-1, h, w)

    keypoints = [
        torch.nonzero(s > config['keypoint_threshold'])
        for s in scores]

    scores = [s1[tuple(k1.t())] for s1, k1 in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints = [torch.flip(k, [1]).float().data for k in keypoints]
    keypoints, scores = list(zip(*[
        remove_borders(k, s, 4, h, w)
        for k, s in zip(keypoints, scores)]))

    if config['max_keypoints'] >= 0:
        keypoints, scores = list(zip(*[
            top_k_keypoints(k, s, config['max_keypoints'])
            for k, s in zip(keypoints, scores)]))

    descriptors = [sample_keypoint_desc(k1[None], d1[None], 8)[0].cpu()
                   for k1, d1 in zip(keypoints, descriptor_pred)]

    keypoints = [k.cpu() for k in keypoints]
    scores = [s.cpu() for s in scores]

    return {
        'keypoints': keypoints,
        'descriptors': descriptors,  # [维度，个数]
        'scores': scores,
    }


# 将两张图像进行拼接
def get_real_kpts_imgs(config, model, inputs):
    refer_detector_pred, refer_descriptor_pred = model(inputs[:, 0:1, :, :])
    query_detector_pred, query_descriptor_pred = model(inputs[:, 1:2, :, :])

    refer_scores = simple_nms(refer_detector_pred, config['nms_radius'])
    query_scores = simple_nms(query_detector_pred, config['nms_radius'])

    b1, _, h1, w1 = refer_detector_pred.shape
    b2, _, h2, w2 = query_detector_pred.shape

    refer_scores = refer_scores.reshape(-1, h1, w1)
    query_scores = query_scores.reshape(-1, h2, w2)

    refer_keypoints = [
        torch.nonzero(s1 > config['keypoint_threshold'])
        for s1 in refer_scores]
    query_keypoints = [
        torch.nonzero(s2 > config['keypoint_threshold'])
        for s2 in query_scores]

    refer_scores = [s1[tuple(k1.t())] for s1, k1 in zip(refer_scores, refer_keypoints)]
    query_scores = [s1[tuple(k1.t())] for s1, k1 in zip(query_scores, query_keypoints)]

    # Discard keypoints near the image borders
    refer_keypoints = [torch.flip(k1, [1]).float().data for k1 in refer_keypoints]
    query_keypoints = [torch.flip(k2, [1]).float().data for k2 in query_keypoints]

    refer_keypoints, refer_scores = list(zip(*[
        remove_borders(k1, s1, 4, h1, w1)
        for k1, s1 in zip(refer_keypoints, refer_scores)]))
    query_keypoints, query_scores = list(zip(*[
        remove_borders(k2, s2, 4, h2, w2)
        for k2, s2 in zip(query_keypoints, query_scores)]))

    if config['max_keypoints'] >= 0:
        refer_keypoints, refer_scores = list(zip(*[
            top_k_keypoints(k1, s1, config['max_keypoints'])
            for k1, s1 in zip(refer_keypoints, refer_scores)]))
        query_keypoints, query_scores = list(zip(*[
            top_k_keypoints(k2, s2, config['max_keypoints'])
            for k2, s2 in zip(query_keypoints, query_scores)]))

    refer_descriptors = [sample_keypoint_desc(k1[None], d1[None], 8)[0].cpu()
                   for k1, d1 in zip(refer_keypoints, refer_descriptor_pred)]

    query_descriptors = [sample_keypoint_desc(k2[None], d2[None], 8)[0].cpu()
                   for k2, d2 in zip(query_keypoints, query_descriptor_pred)]

    refer_keypoints = [k1.cpu() for k1 in refer_keypoints]
    query_keypoints = [k2.cpu() for k2 in query_keypoints]

    refer_scores = [s1.cpu() for s1 in refer_scores]
    query_scores = [s2.cpu() for s2 in query_scores]

    keypoints_pairs = list(zip(refer_keypoints, query_keypoints))
    descriptors_pairs = list(zip(refer_descriptors, query_descriptors))
    scores_pairs = list(zip(refer_scores, query_scores))

    return {
        'keypoints': keypoints_pairs,
        'descriptors': descriptors_pairs,  # 列表，batch size个，[维度，个数]
        'scores': scores_pairs,
    }
