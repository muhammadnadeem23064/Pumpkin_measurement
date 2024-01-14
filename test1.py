import os
import cv2
import numpy as np

img_dir = 'H:\\Practice Codes\\pythonProject\\mask_2.png'

# get list of image file names in directory
#img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

# loop over image files and process each one
#for img_file in img_files:
    # read image
image = cv2.imread(img_dir)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = image_gray.shape
long_array = np.array([])
_, image_v = cv2.threshold(image_gray, 85, 255, cv2.THRESH_TOZERO_INV)
image_blur = cv2.blur(image_v, (3, 3))
long_array = np.array([])
for i in range(0, w):
    for j in range(0, h):
        if 70 <= image_blur[j][i] <= 80:
            image_blur[j][i] = 1
            long_array = np.append(long_array, j)
            print("long array",long_array)
        else:
            image_blur[j][i] = 0
long_array = np.unique(long_array)
long_array.sort()
top_mid = long_array[int(0.2 * len(long_array))]
mid_bot = long_array[int(0.9 * len(long_array))]
top_long = top_mid - np.min(long_array)
mid_long = mid_bot - top_mid
bot_long = np.max(long_array) - mid_bot
long = np.max(long_array) - np.min(long_array)
print('long:{:.2f}cm'.format((long + 1) / 319.0))
print('top_long:{:.2f}cm, mid_long:{:.2f}cm, bottom_long:{:.2f}cm'.format((top_long) / 319.0,(mid_long) / 319.0,(bot_long + 1) / 319.0))