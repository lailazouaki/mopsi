import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import cv2
import tools

sift = cv2.xfeatures2d.SIFT_create()