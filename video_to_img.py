import cv2
import os
import sys
import numpy as np
from get_matches import *
from utils import warp_and_transform, resize_image
from model.SuperRetina.SuperRetina import *
from model.SuperGlue.superglue_kpt import SuperGlue

# 特征提取网络
import yaml
config_path = 'model/ROPconfig.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

feat_config = config['superRetina_params']
match_config = config['train_params']
model_feature = SuperRetina()
checkpoint = torch.load(feat_config['feat_model_path'], map_location=device_list[1])
model_feature.load_state_dict(checkpoint['net'])
model_feature.to(device_list[1])

# 匹配网络
superglue_config = {
    'descriptor_dim': 256,
    'keypoint_encoder': [32, 64, 128, 256],
    'GNN_layers': ['self', 'cross'] * match_config['attention_layers'],
    'sinkhorn_iterations': match_config['sinkhorn_iterations'],
    'match_threshold': match_config['match_threshold']
}
superglue = SuperGlue(config=superglue_config)
superglue = superglue.double()
state_dict = torch.load(match_config['mat_model_path'], map_location=device_list[1])
superglue.load_state_dict(state_dict)
superglue.to(device_list[1])

#######
if __name__ == '__main__':
    # video_path = '/home/yaomm/Data/ROP_Video/ROP_1.avi'
    if len(sys.argv) < 2:
        print("Usage: python process_video.py /path/to/video [homography_type]")
    else:
        video_path = sys.argv[1]
        get_homography = sys.argv[2] if len(sys.argv) == 3 else "get_homography_manual"
        video_filename = os.path.basename(video_path).split('_')[-1].split('.')[0]

        video_capture = cv2.VideoCapture(video_path)
        gt_path = 'Json_path'
        mosaic_path = 'Mosaic_result'
        frame_count = 1
        nums_image = 1

        base_frame = None
        json1_path = None
        result = None
        while True:
            ret, frame = video_capture.read()
            if not ret:
                homo_type = get_homography.split('_')[-1]
                cv2.imwrite(os.path.join(mosaic_path, "frame_{}_mosaic_{}_{}.jpg".format(video_filename, nums_image, homo_type)), result)
                print("finish! There are {} images stitched".format(nums_image))
                break

            # 检查当前帧对应的JSON文件是否存在
            json1_file = f'{gt_path}/frame_{video_filename}_{frame_count:03d}.json'
            if os.path.exists(json1_file):
                if base_frame is None:
                    json1_path = json1_file
                    base_frame = frame.copy()  # 将第一个帧作为基准帧
                    base_frame = resize_image(base_frame)
                    height, width, _ = base_frame.shape
                    result = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
                    start_y = (height * 3 - base_frame.shape[0]) // 2
                    end_y = start_y + base_frame.shape[0]
                    start_x = (width * 3 - base_frame.shape[1]) // 2
                    end_x = start_x + base_frame.shape[1]
                    result[start_y:end_y, start_x:end_x] = base_frame

                else:
                    json2_path = json1_file
                    frame = resize_image(frame)  # resize为1200*1200
                    if get_homography == 'get_homography_manual':
                        pts_src1, pts_dst1 = get_homography_manual(json1_path, json2_path)
                    elif get_homography == 'get_homography_opencv':
                        pts_src1, pts_dst1 = get_homography_opencv(feat_config, model_feature, base_frame, frame)
                    else:
                        pts_src1, pts_dst1 = get_homography_superglue(feat_config, model_feature, superglue, base_frame, frame)

                    if len(pts_src1) < 4:
                        frame_count += 1
                        continue
                    # 创建一个足够大的画布来容纳多个图像的完整区域
                    height, width, _ = frame.shape
                    full_canvas = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
                    warped_image2, H_m = warp_and_transform(full_canvas, frame, pts_src1, pts_dst1)
                    nums_image += 1
                    result = np.maximum(result, warped_image2)

            frame_count += 1

        video_capture.release()
