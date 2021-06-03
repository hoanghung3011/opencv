from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# BUOC 0: Read image
img = cv2.imread('input.jpg', 0)

#img = np.array([[7, 7, 6, 7, 5],[2, 7, 6, 3, 4],[6, 7, 6, 5, 5],[5, 7, 3, 5, 7]])
L = 256
plt.hist(img.ravel(),256,[0,256])

# BUOC 1: Calculate histogram
px = np.zeros(256)
# Cach 1
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        px[img[i,j]] += 1
# Cach 2: flatten image to array
img_arr = img.reshape(-1)
for v in img_arr:
    px[v] += 1
#print(px)
#print('# Cach 2: #pixel of 100: ', px[100])
# Cach 3: for each intensitive value, then count
#for v in range(0, 256):
#    count = np.where(img==100)
#    px[v] = len(count[0])
# Cach 4: dung ham unique cua numpy
#img_arr = img.reshape(-1)
#unique, counts = np.unique(img_arr, return_counts = True)
#for i, val in enumerate(unique):
#    px[val] = counts[i]
#print('Cach 4: #pixel of 100: ', px[100])

# BUOC 2: Calculate CDF and
# BUOC 3: Map intensitive value from original image to equalized one
cdf = 0
cdf_min = np.min(px[px>0])
print('cdf_min: ', cdf_min)
npixel = len(img_arr)
out_img = img.copy()

map_val = {}
for v in range(0,L):
    # Calculate CDF(v
    cdf += px[v]
    # Calculate h(v)
    h_v = np.round((cdf - cdf_min)/(npixel - cdf_min)*(L-1))
    map_val[v] = h_v
    # DONT DO THAT: out_img[out_img==v] = h_v
    out_img[img==v] = h_v

#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        out_img[i,j] = map_val[img[i,j]]

plt.hist(out_img.ravel(),256,[0,256])
plt.show()

cv2.imshow('Original image', img)
cv2.imshow('Equalized image', out_img)
cv2.waitKey()
cv2.destroyAllWindows()