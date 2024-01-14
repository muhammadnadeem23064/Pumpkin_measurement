import numpy as np
import cv2
from PIL import Image, ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
# Load mask image
#mask = Image.open('H:\\Practice Codes\\pythonProject\\mask_1.png')



"""
# Convert the image to a binary array
mask_arr = np.array(mask) > 0

# Find the row-wise and column-wise maximum values
row_max = np.max(mask_arr, axis=1)
col_max = np.max(mask_arr, axis=0)

# Find the minimum and maximum row and column indices
min_row = np.argmax(row_max)
print(min_row)
max_row = len(row_max) - 1 - np.argmax(row_max[::-1])
print(max_row)
min_col = np.argmax(col_max)
print(min_col)
max_col = len(col_max) - 1 - np.argmax(col_max[::-1])

print(max_col)

# Calculate the maximum x1, y1 and x2, y2 coordinates
x1, y1 = min_col, min_row
x2, y2 = max_col, max_row

# Draw a bounding box around the region
draw = ImageDraw.Draw(mask)
cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the image with the bounding box
mask.save('mask_with_bbox.png')
"""
mask = np.array(Image.open('mask_1.png').convert('L'))

# Find the indices of all non-zero elements in the mask
nonzero_indices = np.nonzero(mask)

# Calculate the max and min x and y coordinates
max_x = np.max(nonzero_indices[1])
min_x = np.min(nonzero_indices[1])
max_y = np.max(nonzero_indices[0])
min_y = np.min(nonzero_indices[0])


width= (max_x-min_x)/5.5
height= (max_y-min_y)/9

print("Height",height)
print("width",width)

# Print the results
print('Max X:', max_x)
print('Min X:', min_x)
print('Max Y:', max_y)
print('Min Y:', min_y)