import cv2
import os
import numpy as np
import time
import cPickle as pickle

PATH = "/Users/lailazouaki/Documents/MOPSI/"

#in the following code, K is the branch factor of the vocabulaty tree.
#                       mu is the list of centers for a given cluster of points (divided in K region, hence K centers)
#                       X generally refers to a cluster. A cluster isn't solely made of points, it is made of short list containing. [keypoint,descriptor(which is a vector), img id]
# we call X cluster thanks to the descriptors vector inside it. The centers are computed based on these points(vectors). 
#                       L is the max depth of the voc-Tree.

# for an int K and an int dim, we create K random vectors of dimension dim. 
def random_centers(X, K, dim):
    random_pos = np.random.randint(0, len(X), size = K)
    res = []
    for k in range(0,K):
        res.append(X[random_pos[k]][1])
    return(res)
    
#for a set of elements X     and centers mu (cf previous and new function, create clusters based on nearest center. 
def cluster_points(X, mu):
    clusters  = {}
    for i in range(0, len(mu)):
        clusters[i]=[]
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x[1]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

# for a set of clusters (determined thanks to previous function) we actualize the positions of centers
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(vec_mean(clusters[k]))
    return newmu

#-------------due to the apect of training_data .means is redefined-------------------
#the training dataset is a list of [keypoint,descriptor,associated_image], we are interested in the descriptor which is a vector
# of dimension 64 or 128. Dirty solution
def vec_mean(cluster, dim = 64):
    try:
        sum=[]
        for vector in cluster:
            if(len(sum)==0):
                sum = vector[1]
            else:
                for i in range(0, dim):
                    sum[i]=sum[i]+ vector[1][i]
        for i in range(0, dim):
            sum[i] = sum[i]/len(cluster)
    except IndexError:
        # si jamais il n'y a pas de d'elements dans le cluster considere, le vecteur moyen est mis a 0. 
        # je ne suis absolument pas convaincu, mais n'ai pas trouve d'autre solution. 
        # car il faut absolument remplir les K branches partant de la node, on ne peut pas se permettre d'avoir une branche vide. 
        # Je pense que c'est d'ici qu evient l'erreur d'entrainement. 
        for i in range(0, dim):
            sum.append(0)
    return(sum)

#----------we check the last iteration didn't improve centers location-------------
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 #en gros, la technique d'iteration utilisee pour trouver les centres est asymptotiques.
 # cette condition de convergence va faire office de condition d'arret. 

 #--------- we divide a given area in K parts------------------------------
def find_centers(X, K):
    # Initialize to K random centers
    dim = 64
    oldmu = random_centers(X, K, dim)
    mu = random_centers(X, K, dim)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

    #On boule tant qu'un regime statique n'a pas encore ete atteint. 
    # A chaque fois il s'agit de recalculer la decoupe de X en clusters en utilisant de nouveaux points. 

#------for each image in the training set, we compute how many descriptors there are and return a dictionnary. 

def num_desc(trainingset):
    dic = {}
    for element in trainingset:
        found = False
        for img in dic.keys():
            if(element[2]==img):
                dic[element[2]]= dic[element[2]]+1
                found =True
        if(found == False):
            dic[element[2]]=1
    return dic     

 # Je parcours le trainingset en recencant toutes les photos ainsi qu'en omptant leur nombre de descripteurs.

#----------for a given cluster, finds all images in it an weights them------------------

def set_mass(X, dic):
    masses={}
    for desc in X:
        if desc[2] in masses:
            masses[desc[2]] = masses[desc[2]]+ (float(1)/dic[desc[2]])
        else:
            masses[desc[2]] = (float(1)/dic[desc[2]])
    return masses

# une version assez propres qui ajoute de a valeur a un dico, ou bien cree l'entree. 
# A noter aue le float(1) est la pour forcer une division relle(R), sinon j'obtients tjrs0.


#----------- creates the KDTree, including vectors and masses
def recursive_Tree (X, K, depth, dic, L=2):
    Tree = []
    (mu, clusters) = find_centers(X, K)
    if (depth <= L-2):
        depth = depth+1
        for i in range(0,len(mu)):
            Tree.append([mu[i], recursive_Tree(clusters[i], K, depth, dic)])
        return Tree
    else:
        for i in range(0, len(mu)):
            masses=set_mass(clusters[i], dic)
            Tree.append([mu[i], masses])
        return Tree
#En procedant ainsi, on garde en memoire dans Tree: la hierarchie de l'arbre, les vecteurs aux nodes et les ensembles. 
# Puisque l'arbre est complet on sait exactement la ou tout se trouve.

def save_objectf(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

#------------ reading of training data-----------------
#all pictures have to be in the same folder. 

start_time = time.time()

direction = PATH+"images/image_all"
surf = cv2.xfeatures2d.SURF_create()
trainingset = []
for file in os.listdir(direction):
    img = cv2.imread(direction+"/"+file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (kps,descs) = surf.detectAndCompute(gray, None)
    for j in range(0,len(descs)):
        trainingset.append((kps[j],descs[j],direction+"/"+file))

dictionnary = num_desc(trainingset)

# trainingset is a very large set of lists containing the keypoint, the descriptor and the associated image.


KMTree = recursive_Tree(trainingset, 2, 0, dictionnary)

with open('trained_tree.pkl', 'wb') as output:
    pickle.dump(KMTree, output, -1)

del KMTree

print("Process terminated : done in--- %s seconds --- " %(time.time()-start_time))



