import numpy as np
import cv2
import time


MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# FlannBasedMatcher
def matchFeaturesSIFT(dir1, dir2, dir3):
    start_time = time.time()

    img1 = cv2.imread(dir1, 0)          # queryImage
    img2 = cv2.imread(dir2, 0)          # trainImage

    # initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # here kpX will be a list of keypoints and desX is a numpy array of shape len(kpX) * 128
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print("# SIFT kp1: {}, descriptors1: {}".format(len(kp1), des1.shape))
    print("# SIFT kp2: {}, descriptors2: {}".format(len(kp2), des2.shape))

    # returns k best matches
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imwrite(dir3, img3)
    print("Process terminated : done in--- %s seconds --- " %(time.time()-start_time))
    print(index_params)
    print(search_params)


def matchFeaturesSURF(dir1, dir2, dir3):
    start_time = time.time()

    img1 = cv2.imread(dir1, 0)          # queryImage
    img2 = cv2.imread(dir2, 0)          # trainImage

    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create()
    # print("Initiate SURF detector : done in --- %s seconds --- " %(time.time()-start_time))

    # find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    print("# SURF kp1: {}, descriptors1: {}".format(len(kp1), des1.shape))
    print("# SURF kp2: {}, descriptors2: {}".format(len(kp2), des2.shape))

    matches = flann.knnMatch(des1,des2,k=2)
    # print("Keypoints and descriptors found : done in--- %s seconds --- " %(time.time()-start_time))

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imwrite(dir3, img3)
    print("Process terminated : done in--- %s seconds --- " %(time.time()-start_time))


# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 30, # 12
#                    key_size = 8,     # 20
#                    multi_probe_level = 2) #2


# # BruteForceMatcher
# def BFMatchFeaturesSIFT(dir1, dir2, dir3):
#     start_time = time.time()

#     img1 = cv2.imread(dir1, 0)          # queryImage
#     img2 = cv2.imread(dir2, 0)          # trainImage

#     # Initiate SIFT detector
#     sift = cv2.xfeatures2d.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(img1,None)
#     kp2, des2 = sift.detectAndCompute(img2,None)

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1,des2, k=2)

#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m,n in matches:
#         if m.distance < 0.7*n.distance:
#             good.append(m)


#     if len(good)>MIN_MATCH_COUNT:
#         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#         matchesMask = mask.ravel().tolist()

#         h,w = img1.shape
#         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#         dst = cv2.perspectiveTransform(pts,M)

#         img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#     else:
#         print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#         matchesMask = None


#     draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                        singlePointColor = None,
#                        matchesMask = matchesMask, # draw only inliers
#                        flags = 2)

#     img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#     cv2.imwrite(dir3, img3)
#     print("--- %s seconds --- " %(time.time()-start_time))

# def BFMatchFeaturesSURF(dir1, dir2, dir3):
#     start_time = time.time()

#     img1 = cv2.imread(dir1, 0)          # queryImage
#     img2 = cv2.imread(dir2, 0)          # trainImage

#     # Initiate SIFT detector
#     surf = cv2.xfeatures2d.SURF_create()

#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = surf.detectAndCompute(img1,None)
#     kp2, des2 = surf.detectAndCompute(img2,None)

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1,des2, k=2)

#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m,n in matches:
#         if m.distance < 0.7*n.distance:
#             good.append(m)


#     if len(good)>MIN_MATCH_COUNT:
#         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#         matchesMask = mask.ravel().tolist()

#         h,w = img1.shape
#         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#         dst = cv2.perspectiveTransform(pts,M)

#         img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

#     else:
#         print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#         matchesMask = None


#     draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                        singlePointColor = None,
#                        matchesMask = matchesMask, # draw only inliers
#                        flags = 2)

#     img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#     cv2.imwrite(dir3, img3)
#     print("--- %s seconds --- " %(time.time()-start_time))