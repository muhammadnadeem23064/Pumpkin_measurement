from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2
import numpy as np
from PIL import Image

image = cv2.imread('C:/Users/Taicheng/Desktop/12.jpg')
image1 = Image.open('C:/Users/Taicheng/Desktop/02.jpg')

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = image_gray.shape
long_array = np.array([])
_, image_v = cv2.threshold(image_gray, 85, 255, cv2.THRESH_TOZERO_INV)
image_blur = cv2.blur(image_v, (3, 3))
long_array = np.array([])
total_red = 0
total_green = 0
total_blue = 0
num_pixels = 0
for i in range(0, w):
    for j in range(0, h):
        if image_blur[j][i] >= 70 and image_blur[j][i] <= 80:
            image_blur[j][i] = 1
            long_array = np.append(long_array, j)
        else:
            image_blur[j][i] = 0
skeleton = skeletonize(image_blur)

for i in range(0, w):
    for j in range(0, h):
        if  skeleton[j][i] == True:
            pixel = image1.getpixel((i, j))
            total_red += pixel[0]
            total_green += pixel[1]
            total_blue += pixel[2]
            num_pixels += 1
long_array = np.unique(long_array)
long_array.sort()
top_mid = long_array[int(0.2 * len(long_array))]
mid_bot = long_array[int(0.9 * len(long_array))]
top_long = top_mid - np.min(long_array)
mid_long = mid_bot - top_mid
bot_long = np.max(long_array) - mid_bot
long = np.max(long_array) - np.min(long_array)


wide_array = np.array([])
top_wide_array = np.array([])
mid_wide_array = np.array([])
bot_wide_array = np.array([])

for j in long_array.astype(int):
    if j < top_mid:
        for i in range(0, w):
            if image_blur[j][i] == 1:
                wide_array = np.append(wide_array, i)
        wide1 = np.max(wide_array) - np.min(wide_array)
        top_wide_array = np.append(top_wide_array, wide1)
        wide_array = []
    elif j >= top_mid and j < mid_bot:
        for i in range(0, w):
            if image_blur[j][i] == 1:
                wide_array = np.append(wide_array, i)

        wide2 = np.max(wide_array) - np.min(wide_array)
        mid_wide_array = np.append(mid_wide_array, wide2)
        wide_array = []
    else:
        for i in range(0, w):
            if image_blur[j][i] == 1:
                wide_array = np.append(wide_array, i)

        wide3 = np.max(wide_array) - np.min(wide_array)
        bot_wide_array = np.append(bot_wide_array, wide3)
        wide_array = []
wide1 = ((np.max(top_wide_array)) + 1) / 319.0
wide2 = ((np.max(mid_wide_array)) + 1) / 319.0
wide3 = ((np.max(bot_wide_array)) + 1) / 319.0
avg_red = total_red // num_pixels
avg_green = total_green // num_pixels
avg_blue = total_blue // num_pixels
print('long:{:.2f}cm'.format((long + 1) / 319.0))
print('top_long:{:.2f}cm, mid_long:{:.2f}cm, bottom_long:{:.2f}cm'.format((top_long) / 319.0,(mid_long) / 319.0,(bot_long + 1) / 319.0))
print('top_wide:{:.2f}cm, mid_wide:{:.2f}cm, bottom_wide:{:.2f}cm'.format(wide1, wide2, wide3))
print("RGB ({}, {}, {})".format(avg_red, avg_green, avg_blue))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image1, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('edge_skeleton', fontsize=20)

# ax[0].imshow(cv2.addWeighted(image_leaves, 1, image_vege, 1, 0), cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('original', fontsize=20)
#
# ax[1].imshow(cv2.addWeighted(image_edge_l, 1, image_edge, 1, 0), cmap=plt.cm.gray)
# ax[1].axis('off')
# ax[1].set_title('edge', fontsize=20)

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('skeleton', fontsize=20)

ax[1].imshow(image_blur, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('final_image', fontsize=20)
# plt.text(-2,-2,str(long),fontsize=16,colro='red')

fig.tight_layout()
plt.show()