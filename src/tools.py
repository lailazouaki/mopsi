import numpy as np
import cv2


def loadImages(dir1, dir2):
    img1 = cv2.imread(dir1, 0)          # queryImage
    img2 = cv2.imread(dir2, 0)          # trainImage
    return img1, img2

def extractKeypointsDescriptors(img, detector):
    kp, des = detector.detectAndCompute(img,None)
    print("# SIFT kp1: {}, descriptors1: {}".format(len(kp), des.shape))

