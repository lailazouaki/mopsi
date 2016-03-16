import cv2
import numpy as np
import os

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


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(vec_mean(clusters[k]))
    return newmu

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
        for i in range(0, dim):
            sum.append(0)
    return(sum)

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

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

trainingset=[]
direction = "/Users/Thomartin/mopsi/images/tour_eiffel"
surf = cv2.xfeatures2d.SURF_create()
trainingset = []
for file in os.listdir(direction):
    img = cv2.imread(direction+"/"+file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (kps,descs) = surf.detectAndCompute(gray, None)
    for j in range(0,len(descs)):
        trainingset.append((kps[j],descs[j],direction+"/"+file))

print(type(trainingset[4][1].tolist()[1]))
cen = random_centers(4, 64)
clus = cluster_points(trainingset, cen)
# print(reevaluate_centers(cen, clus ))
# print(cen)
# print(clus.keys())
set

