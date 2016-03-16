import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import featureMatching
import cPickle as pickle


with open('trained_tree.pkl', 'rb') as input_tree:
	KMTree = pickle.load(input_tree)

# print(KMTree)
print(KMTree[0][1][1][1]['/Users/lailazouaki/Documents/MOPSI/images/image_all/tour_eiffel_2.jpg'])