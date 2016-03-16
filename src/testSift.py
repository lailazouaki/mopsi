import cv2
import numpy as np
import os

img = cv2.imread("/Users/Thomartin/mopsi/images/tour_eiffel/tour_eiffel_3.jpg")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

img=cv2.drawKeypoints(gray,kps, img)

cv2.imwrite('sift_keypoints.jpg',img)








