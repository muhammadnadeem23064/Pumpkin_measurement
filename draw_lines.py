import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the binary mask image
mask = cv.imread("H:\\Practice Codes\\pythonProject\\mask_0.png", cv.IMREAD_GRAYSCALE)
ret, thresh = cv.threshold(mask, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

area = cv.contourArea(cnt)

perimeter = cv.arcLength(cnt, True)

epsilon = 0.1 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

hull = cv.convexHull(cnt)

k = cv.isContourConvex(cnt)

x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

# rotate rectangel

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(mask, [box], 0, (0, 0, 255), 2)

"""
#draw vertical line
rows, cols = mask.shape[:2]
[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv.line(mask, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
plt.imshow(mask)
plt.show()
"""


rows, cols = mask.shape[:2]

# Calculate the line parameters using fitLine
[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)

if vy != 0:
    # Calculate the slope of the line and its y-intercept
    slope = vx / vy
    y_intercept = y - slope * x

    # Calculate the y-coordinate of the horizontal line
    y_coord = int(rows / 2)
    print(y_coord)

    # Calculate the x-coordinate of the horizontal line at the maximum x value
    line_end = np.argmax(mask[:, 0])  # Find the index of the maximum x value
    x_coord = line_end % cols

    # Draw the horizontal line on the mask image
    cv.line(mask, (0, y_coord), (x_coord, y_coord), (0, 255, 0), 25)

    # Calculate the width of the line and print it
    line_points = cv.findNonZero(mask)
    print(line_points)
    x_coords = line_points[:, 0, 0]  # Extract x-coordinates of all points on the line
    min_x = min(x_coords)
    max_x = max(x_coords)
    width = max_x - min_x + 1  # Add 1 to account for 0-based indexing
    print("Width of the line:", max_x)


# Display the image
plt.imshow(mask)
plt.show()










