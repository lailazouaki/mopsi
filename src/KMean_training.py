import cv2
import os
import numpy as np


def random_centers(K, dim):
    res = []
    for k in range(0,K):
        vec=[]
        for i in range(0,dim):
            vec.append(random.uniform(-1, 1))
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
    for cluster in clusters
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
    oldmu = random_centers(K, dim)
    mu = random_centers(K, dim)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)



#----------for a given cluster, finds all images in it an weights them------------------
def set_mass(X):
    masses = []
    for desc in X:
        found = false
        for i in range(0, len(masses)):
            if(masses[i][1]==desc[2]):
                masses[i][0]=masses[i][0] + 1
                found = true
        if(found == false):
            masses.append([1, desc[2]])
    return masses


#----------- creates the KDTree, including vectors and masses
def recursive_Tree (X, K, depth):
    if(depth =< L):
        (mu, clusters) = find_centers(X, K)
        depth = depth-1
        for X in clusters:
            Tree[i]=[mu[i], recursive_Tree(X, K, depth)]
    return Tree
    else:
        for i in range(0, len(mu)):
            masses=set_mass(X)
            Tree[i]=[mu[i], masses]
        return Tree
#En procedant ainsi, on garde en memoire dans Tree: la hierarchie de l'arbre, les vecteurs aux nodes et les ensembles. 
# Puisque l'arbre est complet on sait exactement là ou tout se trouve.


#------------ reading of training data-----------------
#all pictures have to be in the same folder. 

direction = "/Users/Thomartin/mopsi/images/tour_eiffel/"
surf = cv2.xfeatures2d.SURF_create()
trainingset = []
for i in os.listdir(direction):
    img = cv2.imread(direction+i)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (kps,descs) = surf.detectAndCompute(gray, None)
    for j in range(0,len(descs)):
        trainingset.append((kps[j],descs[j],direction+i))
# trainingset is a very large set of lists containing the keypoint, the descriptor and the associated image.

KMTree = recursive_Tree(trainingset, K, 0)

