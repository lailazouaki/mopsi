from os import listdir
from os.path import isfile, join
import sys
import featureMatching


print("---- Feature Matching ----")

WHOLE_PATH_IMAGES = "/Users/Thomartin/mopsi/images/tour_eiffel/"

images = [WHOLE_PATH_IMAGES+sys.argv[1]+"/"+f for f in listdir(WHOLE_PATH_IMAGES+sys.argv[1]) if (not f.startswith('.') and isfile(join(WHOLE_PATH_IMAGES+sys.argv[1], f)))]

for image in images:
	print(image)

# SIFT - Flann Matcher
for count, image in enumerate(images):
	featureMatching.matchFeaturesSIFT(images[0], image, "../match/"+sys.argv[1]+"/sift/match_image_"+str(count)+".jpg")

# # SURF - Flann Matcher
# for count, image in enumerate(images):
#     featureMatching.matchFeaturesSURF(images[0], image, "../match/"+sys.argv[1]+"/surf/match_image_"+str(count)+".jpg")

# # SIFT - BF Matcher 
# for count, image in enumerate(images):
#     featureMatching.BFMatchFeaturesSIFT(images[0], image, "../match/sift/"+sys.argv[1]+"/match_image_"+str(count)+".jpg")
