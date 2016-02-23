from os import listdir
from os.path import isfile, join
import sys
import featureMatching

print("---- Feature Matching ----")

WHOLE_PATH_IMAGES = "/Users/lailazouaki/Documents/MOPSI/images/"
images = [WHOLE_PATH_IMAGES+sys.argv[1]+"/"+f for f in listdir(WHOLE_PATH_IMAGES+sys.argv[1]) if (not f.startswith('.') and isfile(join(WHOLE_PATH_IMAGES+sys.argv[1], f)))]

for image in images:
	print(image)

for count, image in enumerate(images):
    featureMatching.matchFeatures(images[0], image, "../match/"+sys.argv[1]+"/match_image_"+str(count)+".jpg")

