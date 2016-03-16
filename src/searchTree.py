import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import featureMatching
import cPickle as pickle


def find_nearest_center(List_SubTree, descriptor):
    return min([(i, np.linalg.norm(descriptor-List_SubTree[i][0])) \
                    for i in enumerate(List_SubTree)], key=lambda t:t[1])[0]


def getDicMass(KMTree, descriptor):
    Tree = KMTree
    for i in range(0, L):
        branchNumber = find_nearest_center(Tree, descriptor)
        Tree = Tree[branchNumber][1]
    return Tree 


with open('trained_tree.pkl', 'rb') as input_tree:
	KMTree = pickle.load(input_tree)

# print(KMTree)
print(KMTree[0][1][1][1]['/Users/lailazouaki/Documents/MOPSI/images/image_all/tour_eiffel_2.jpg'])



PATH = "/Users/lailazouaki/Documents/MOPSI/images/"
# img_path =PATH+sys.argv[1]

surf = cv2.xfeatures2d.SURF_create()
gray = cv2.imread("/Users/lailazouaki/Documents/MOPSI/images/ndp_7.jpg", 0)
(kps,descs) = surf.detectAndCompute(gray, None)

relevance_database={}
for img in os.listdir(PATH+"image_all"):
    relevance_database[PATH+"image_all/"+img] =0

test = getDicMass(KMTree, descs[0])
print(test)
