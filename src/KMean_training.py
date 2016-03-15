import cv2
import os
import numpy as np

#in the following code, K is the branch factor of the vocabulaty tree.
#                       mu is the list of centers for a given cluster of points (divided in K region, hence K centers)
#                       X generally refers to a cluster. A cluster isn't solely made of points, it is made of short list containing. [keypoint,descriptor(which is a vector), img id]
# we call X cluster thanks to the descriptors vector inside it. The centers are computed based on these points(vectors). 
#                       L is the max depth of the voc-Tree.

def random_centers(K, dim):
    res = []
    for k in range(0,K):
        vec=[]
        for i in range(0,dim):
            vec.append(np.random.uniform(-1, 1))
        res.append(vec)
    return(res)
    
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x[1]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    # keys = sorted(clusters.keys())
    # for k in keys:
    for cluster in clusters:
        newmu.append(vec_mean(cluster))
    return newmu

#-------------due to the apect of training_data .means is redefined-------------------
#the training dataset is a list of [keypoint,descriptor,associated_image], we are interested in the descriptor which is a vector
# of dimension 64 or 128. Dirty solution
def vec_mean(cluster):
    sum=[]
    for vector in cluster:
        if(len(sum)==0):
            sum = vector
        else:
            for i in range(0, dim):
                sum[i]=sum[i]+vector[1][i]
    for i in range(0,dim):
        sum[i] = sum[i]/len(cluster)
    return(sum)

#----------we check the last iteration didn't improve centers location-------------
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
 #--------- we divide a given area in K parts------------------------------
def find_centers(X, K):
    # Initialize to K random centers
    dim = 64
    oldmu = random_centers(K, dim)
    mu = random_centers(K, dim)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

#------for each image in the training set, we compute how many descriptors there are and return a dictionnary. 

def num_desc(trainingset):
    dic = {}
    for element in trainingset:
        found = False
        for img in dic.keys():
            if(found==False):
                dic[img] = 1
            if(element[2]==img):
                dic[img]= dic[img]+1
                found =True     
    return dic     



#----------for a given cluster, finds all images in it an weights them------------------
def set_mass(X, dic):
    masses = []
    for desc in X:
        found = False
        print(len(masses))
        for i in range(0, len(masses)):
            print(masses[i][1])
            if(masses[i][1]==desc[2]):
                masses[i][0]=masses[i][0] + 1/dic[masses[i][1]]
                found = False
        if(found == False):
            masses.append([1/dic[desc[2]]])
    return masses


#----------- creates the KDTree, including vectors and masses
def recursive_Tree (X, K, depth, dic, L=5):
    if depth <= L:
        (mu, clusters) = find_centers(X, K)
        depth = depth-1
        for X in clusters:
            Tree[i]=[mu[i], recursive_Tree(X, K, depth, dic)]
        return Tree
    else:
        for i in range(0, len(mu)):
            masses=set_mass(X, dic)
            Tree[i]=[mu[i], masses]
        return Tree
#En procedant ainsi, on garde en memoire dans Tree: la hierarchie de l'arbre, les vecteurs aux nodes et les ensembles. 
# Puisque l'arbre est complet on sait exactement la ou tout se trouve.


#------------ reading of training data-----------------
#all pictures have to be in the same folder. 

direction = "/Users/lailazouaki/Documents/MOPSI/images/tour_eiffel/"
surf = cv2.xfeatures2d.SURF_create()
trainingset = []
for i in os.listdir(direction):
    print(i)
    gray = cv2.imread(direction+str(i), 0)
    (kps,descs) = surf.detectAndCompute(gray, None)
    for j in range(0,len(descs)):
        trainingset.append((kps[j],descs[j],direction+i))
dictionnary = num_desc(trainingset)
# trainingset is a very large set of lists containing the keypoint, the descriptor and the associated image.

KMTree = recursive_Tree(trainingset, 2, 0, dictionnary)

