import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import featureMatching

img = cv2.imread(sys.argv[1])
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)

for vector in descs:

print(len(kps))
print(len(descs))