from os import listdir
from os.path import isfile, join

import cv2
import sys
import featureMatching
import operator 

import numpy as np
import cPickle as pickle


PATH = "/Users/lailazouaki/Documents/MOPSI/images/"

def openObject(filename):
	with open(filename, 'rb') as obj:
		openedObject = pickle.load(obj)

	return openedObject
<<<<<<< HEAD


def find_nearest_center(list_subTree, descriptor):
    return min([(i, np.linalg.norm(descriptor-list_subTree[i][0])) \
                    for i, subtree in enumerate(list_subTree)], key=lambda t:t[1])[0]

=======
>>>>>>> Nettoyage et réarrangement du code. Possibilité d'enregistrer les arbres entrainés. Tests avec différents L, K et taille d'image database

def getDicScore(trained_tree, descriptor, max_depth=2):
    tree = trained_tree
    for i in range(0, max_depth):
        branchNumber = find_nearest_center(tree, descriptor)
        tree = tree[branchNumber][1]

<<<<<<< HEAD
=======
def find_nearest_center(list_subTree, descriptor):
    return min([(i, np.linalg.norm(descriptor-list_subTree[i][0])) \
                    for i, subtree in enumerate(list_subTree)], key=lambda t:t[1])[0]


def getDicScore(trained_tree, descriptor, max_depth=2):
    tree = trained_tree
    for i in range(0, max_depth):
        branchNumber = find_nearest_center(tree, descriptor)
        tree = tree[branchNumber][1]

>>>>>>> Nettoyage et réarrangement du code. Possibilité d'enregistrer les arbres entrainés. Tests avec différents L, K et taille d'image database
    return tree 

# Score de toutes les images pour un descripteur
def incrementDic(relevance_database, dicScore):
	for img_name in dicScore.keys():
		relevance_database[img_name] += dicScore[img_name]

# Score de toutes les images pour tous les descripteurs = pour la query image
def incrementTotal(relevance_database, trained_tree, decs):
	for descriptor in descs:
		dicScore = getDicScore(trained_tree, descriptor)
		incrementDic(relevance_database, dicScore)


if __name__ == "__main__":
		
	trained_tree = openObject('trained_tree.pkl')
<<<<<<< HEAD

	surf = cv2.xfeatures2d.SURF_create()
	gray = cv2.imread("/Users/lailazouaki/Documents/MOPSI/images/ndp_7.jpg", 0)
	(kps,descs) = surf.detectAndCompute(gray, None)

	# Initialisation du dictionnaire de pertinence des images de la base de donnees
	relevance_database={}
	for img in listdir("/Users/lailazouaki/Documents/MOPSI/images/image_all"):
	    relevance_database["/Users/lailazouaki/Documents/MOPSI/images/image_all/"+img] = 0


	# Calcul du score des images de la base de donnees pour la query image
	incrementTotal(relevance_database, trained_tree, descs)
	print(relevance_database)
	print(max(relevance_database.values()))
	print(relevance_database["/Users/lailazouaki/Documents/MOPSI/images/image_all/ndp_7.jpg"])
=======
	print(trained_tree)
	surf = cv2.xfeatures2d.SURF_create()
	gray = cv2.imread("/Users/lailazouaki/Documents/MOPSI/images/ndp_1.jpg", 0)
	(kps,descs) = surf.detectAndCompute(gray, None)

	# Initialisation du dictionnaire de pertinence des images de la base de donnees
	relevance_database={}
	for img in listdir("/Users/lailazouaki/Documents/MOPSI/images/image_all"):
	    relevance_database["/Users/lailazouaki/Documents/MOPSI/images/image_all/"+img] = 0


	# Calcul du score des images de la base de donnees pour la query image
	incrementTotal(relevance_database, trained_tree, descs)
	# sorted_relevance = sorted(relevance_database.items(), key=operator.itemgetter(1))

	print(relevance_database.values())
	print("-----------------------")
	print(max(relevance_database.values()))
	# for element in reversed(sorted_relevance):
	# 	print(element)
>>>>>>> Nettoyage et réarrangement du code. Possibilité d'enregistrer les arbres entrainés. Tests avec différents L, K et taille d'image database



# Trash but maybe useful later

	# dicScore = getDicScore(trained_tree, descs[0])
	# print(len(dicScore.keys()))

	# incrementDic(relevance_database, dicScore)

	# for key in dicScore:
	# 	print(str(relevance_database[key]) + " -- " + str(dicScore[key])) 





