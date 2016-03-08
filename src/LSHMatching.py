import numpy as np
import cv2

img1 = cv2.imread(dir1, 0)          # queryImage
img2 = cv2.imread(dir2, 0)          # trainImage

# Find the keypoints and descriptors with SURF
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
