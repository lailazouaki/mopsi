import cv2
import numpy as np
import os

for i in os.listdir("/Users/Thomartin/mopsi/images/tour_eiffel/"):
    img = cv2.imread("/Users/Thomartin/mopsi/images/tour_eiffel/"+i)
    print(img)
