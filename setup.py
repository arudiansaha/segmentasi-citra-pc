import cv2 as cv

img = cv.imread('img/owl_100.jpg')

# rescale img frame
widht = int(img.shape[1] * 0.25)
height = int(img.shape[0] * 0.25)

img_resized = cv.resize(img, (widht, height), interpolation = cv.INTER_AREA)
cv.imshow('Owl', img_resized)

cv.waitKey(0)
