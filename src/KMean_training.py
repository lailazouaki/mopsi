import cv2
import os
import time
import sys
import numpy as np
import cPickle as pickle


PATH = "/Users/lailazouaki/Documents/MOPSI/"

# In the following code:
# - K is the branch factor of the vocabulaty tree.
# - cluster_centers is the list of centers for a given cluster of points (divided in K region, hence K centers)
# - X generally refers to a cluster. A cluster isn't solely made of points. It is made of short list containing : [keypoint, descriptor, img id]
# - descriptor is a 64-dim vector 
# We call X cluster because of the descriptors vector inside it. The centers are computed based on these points (vectors). 
# - L is the max depth of the voc-Tree.

# We create K random vectors of dimension dim. 
def random_centers(X, K, dim):

    random_pos = np.random.randint(0, len(X), size = K)
    res = [] # A renommer
    for k in range(0,K):
        res.append(X[random_pos[k]][1])

    return(res)
    
# For a set of elements X and centers cluster_centers (determined thanks to previous function), create clusters based on nearest center. 
def cluster_points(X, cluster_centers):

    clusters  = {}
    for i in range(0, len(cluster_centers)):
        clusters[i]=[]
    for x in X:
        best_cluster_center_key = min([(i[0], np.linalg.norm(x[1]-cluster_centers[i[0]])) \
                    for i in enumerate(cluster_centers)], key=lambda t:t[1])[0]
        try:
            clusters[best_cluster_center_key].append(x)
        except KeyError:
            clusters[best_cluster_center_key] = [x]

    return clusters

# For a set of clusters (determined thanks to previous function) we actualize the positions of centers
def reevaluate_centers(cluster_centers, clusters):

    new_cluster_center = []
    keys = sorted(clusters.keys())
    for k in keys:
        new_cluster_center.append(vec_mean(clusters[k]))

    return new_cluster_center

#------------- due to the apect of training_data .means is redefined -------------------
# The training dataset is a list of [keypoint,descriptor,associated_image], we are interested in the descriptor which is a vector
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
        # Si jamais il n'y a pas de d'elements dans le cluster considere, le vecteur moyen est mis a 0. 
        # je ne suis absolument pas convaincu, mais n'ai pas trouve d'autre solution. 
        # car il faut absolument remplir les K branches partant de la node, on ne peut pas se permettre d'avoir une branche vide. 
        # Je pense que c'est d'ici qu evient l'erreur d'entrainement. 
        for i in range(0, dim):
            sum.append(0)

    return(sum)

# ---------- We check the last iteration didn't improve centers location -------------
    
    # La technique d'iteration utilisee pour trouver les centres est asymptotiques.
    # Cette condition de convergence va faire office de condition d'arret.     

def has_converged(cluster_centers, old_cluster_center):
    return (set([tuple(a) for a in cluster_centers]) == set([tuple(a) for a in old_cluster_center]))

#----------- We divide a given area in K parts ------------------------------

    # On boucle tant qu'un regime statique n'a pas encore ete atteint. 
    # A chaque fois il s'agit de recalculer la decoupe de X en clusters en utilisant de nouveaux points. 

def find_centers(X, K):

    # Initialize to K random centers
    dim = 64
    old_cluster_center = random_centers(X, K, dim)
    cluster_centers = random_centers(X, K, dim)
    while not has_converged(cluster_centers, old_cluster_center):
        old_cluster_center = cluster_centers
        # Assign all points in X to clusters
        clusters = cluster_points(X, cluster_centers)
        # Reevaluate centers
        cluster_centers = reevaluate_centers(old_cluster_center, clusters)

    return(cluster_centers, clusters)


#------ For each image in the training set, we compute how many descriptors there are and return a dictionary. 
# Par exemple, dic[img1] = nb_desc_(img1)
def num_desc(X):

    descriptor_count = {}
    for element in X:
        found = False
        for img in descriptor_count.keys():
            if(element[2]==img):
                descriptor_count[element[2]]= descriptor_count[element[2]]+1
                found =True
        if(found == False):
            descriptor_count[element[2]]=1

    return descriptor_count

#---------- For a given cluster, finds all images in it and weights them ------------------

def set_mass(X, descriptor_count_dictionary):

    masses={}
    for desc in X: # A renommer
        if desc[2] in masses: 
            masses[desc[2]] = masses[desc[2]]+ (float(1)/descriptor_count_dictionary[desc[2]])
        else:
            masses[desc[2]] = (float(1)/descriptor_count_dictionary[desc[2]])

    return masses

# A noter que le float(1) est la pour forcer une division reelle, sinon j'obtients toujours 0.


#----------- creates the KDTree, including vectors and masses
def recursive_tree (X, K, current_depth, descriptor_count_dictionary, max_depth=2):

    tree = []
    (cluster_centers, clusters) = find_centers(X, K)

    if (current_depth <= max_depth-2):
        current_depth = current_depth+1
        for i in range(0,len(cluster_centers)):
            tree.append([cluster_centers[i], recursive_tree(clusters[i], K, current_depth, descriptor_count_dictionary)])
        return tree
    else:
        for i in range(0, len(cluster_centers)):
            masses=set_mass(clusters[i], descriptor_count_dictionary)
            tree.append([cluster_centers[i], masses])
        return tree
# En procedant ainsi, on garde en memoire dans tree: la hierarchie de l'arbre, les vecteurs aux nodes et les ensembles. 
# Puisque l'arbre est complet on sait exactement ou tout se trouve.

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

    del obj

#------------ reading of training data-----------------
#all pictures have to be in the same folder. 

if __name__ == "__main__":

    # sys.argv[1] = K
    # sys.argv[2] = L
    # sys.argv[3] = mode : large or small database
    # sys.argv[4] = i-eme execution

    start_time = time.time()

    if str(sys.argv[3]) == "large":
        direction = PATH+"images/image_all"

    else:
        direction = PATH+"images/image_small"

    surf = cv2.xfeatures2d.SURF_create()
    trainingset = []

    for file in os.listdir(direction):
        img = cv2.imread(direction+"/"+file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (kps,descs) = surf.detectAndCompute(gray, None)
        for j in range(0,len(descs)):
            trainingset.append((kps[j],descs[j],direction+"/"+file))

    # trainingset is a very large set of lists containing the keypoint, the descriptor and the associated image.
    descriptor_count_dictionary = num_desc(trainingset)

    # Training of the tree
    trained_tree = recursive_tree(trainingset, int(sys.argv[1]), 0, descriptor_count_dictionary, int(sys.argv[2]))

    # Save trained tree
    save_object(trained_tree, 'trained_tree_'+str(sys.argv[4])+'.pkl')

    # Remind time needed
    with open('trained_tree_'+str(sys.argv[4])+'.txt', 'w') as f:
        f.write("trained_tree_"+sys.argv[4])
        f.write("\nProcess terminated : done in--- %s seconds --- " %(time.time()-start_time))
        f.write("\nK = " + sys.argv[1])
        f.write("\nL = " + sys.argv[2])
        f.write("\nMode = " + sys.argv[3])

    print("Process terminated : done in--- %s seconds --- " %(time.time()-start_time))



