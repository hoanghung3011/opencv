import cv2
import numpy as np

img1 = cv2.imread('mask.png')
img2 = cv2.imread('fire.jpg')
img3 = cv2.imread('girl.jpg')

mask = img1[:, :, 2]

mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

mask_2 = cv2.threshold(img3, 25, 255, cv2.THRESH_BINARY)[1]

h, w = mask_2.shape[:2]
mask_data = np.zeros((h, w))

for k in range(h):
    for l in range(w):
        if np.all(mask_2[k][l] == 0):
            mask_2[k][l] = 255
        elif np.all(mask_2[k][l] > 0):
            mask_2[k][l] = 0

mask_copy = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# db = cv2.bitwise_and(img2, mask_2)

db = cv2.bitwise_or(mask_copy, mask_2) 

db_2 = cv2.bitwise_and(img2, db)

dst = cv2.addWeighted(img3, 0.5, db_2, 0.5, 0)

cv2.imshow('result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
