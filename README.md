## If you want to concatenate the videos, run the following code:
```
python video_to_img.py /path/to/video [homography_type]
```
The 'homography_type' parameter has three options: 'get_homography_manual', 'get_homography_opencv', and 'get_homography_superglue'. The default is 'get_homography_manual'.

'get_homography_manual' calculates the homography matrix using manually annotated matching points.
'get_homography_opencv' calculates the homography matrix by extracting feature points using SuperRetina and obtaining matching point pairs through OpenCV.
'get_homography_superglue' computes the homography matrix using feature points extracted by SuperRetina and matching point pairs obtained via Superglue.

For example,
```
python video_to_img.py ROP_1.avi get_homography_manual
```
The result of the Mosaic using the manually labeled feature points to calculate the homography matrix is as follows:
![image](https://github.com/ymmzs/Video_Stitch/blob/master/Mosaic_result/frame_1_mosaic_50_manual.jpg)
![image](https://github.com/ymmzs/Video_Stitch/blob/master/Mosaic_result/frame_3_mosaic_39_manual.jpg)
