import numpy as np
import cv2


def loadImages(dir1, dir2):
    img1 = cv2.imread(dir1, 0)          # queryImage
    img2 = cv2.imread(dir2, 0)          # trainImage
    return img1, img2

def SIFTDescription(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)