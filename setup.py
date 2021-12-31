import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('img/rock.jpg')

color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# using global threshold to make sure between bg and fg
ret, thresh = cv.threshold(gray, 212, 255, cv.THRESH_BINARY_INV)

# cleaning some noises
kernel = np.ones((6, 6), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 5)

# to make sure where the bg with dilate
bg = cv.dilate(opening, kernel, iterations = 8)

# same as bg but with erode
fg = cv.erode(opening, kernel, iterations = 3)
ret, fg = cv.threshold(fg, 0.9 * fg.max(), 255, 0)
fg = np.uint8(fg)

# to know unknown markers
unknown = cv.subtract(bg, fg)

ret, markers = cv.connectedComponents(fg)
markers = markers + 10
markers[unknown == 255] = 1

# whatershed method
markers = cv.watershed(color, markers)
color[markers == 1] = [255, 0, 0]

plt.figure('Original')
plt.imshow(color)

plt.figure('Grayscale')
plt.imshow(gray, cmap='gray')

plt.figure('Binary')
plt.imshow(thresh, cmap='gray')

plt.figure('Noises Cleaned')
plt.imshow(opening, cmap='gray')

plt.figure('Background')
plt.imshow(bg, cmap='gray')

plt.figure('Foreground')
plt.imshow(fg, cmap='gray')

plt.figure('Unknown')
plt.imshow(unknown, cmap='gray')

plt.figure('Markers')
plt.imshow(markers, cmap='gray')
plt.show()
