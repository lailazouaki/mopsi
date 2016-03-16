import cv2
import numpy as np

img = cv2.imread("/Users/Thomartin/mopsi/images/tour_eiffel/tour_eiffel_2.jpg")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
print(descs[2])
print(len(descs))
img=cv2.drawKeypoints(gray,kps, img)

cv2.imwrite('surf_keypoints.jpg',img)