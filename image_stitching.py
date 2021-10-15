import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

img_ = cv2.imread("original_image_right.jpg")
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread("original_image_left.jpg")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("original_image_left",img)
cv2.waitKey(0)

cv2.imshow("original_image_right",img_)
cv2.waitKey(0)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img,kp1,None))
cv2.waitKey(0)

cv2.imshow('original_image_right_keypoints',cv2.drawKeypoints(img_,kp1,None))
cv2.waitKey(0)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.04*m[1].distance:
       good.append(m)
matches = np.asarray(good)


img3 = cv2.drawMatchesKnn(img,kp1,img_,kp2,matches,None,flags=2)
cv2.imshow("original_image_drawMatches.jpg", img3)
cv2.waitKey(0)

if len(matches[:,0]) >= 4:
     src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
     dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print (H)
else:
     print("Not enought matches are found - %d/%d")

dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0], 0:img.shape[1]] = img
#plt.subplot(122),plt.imshow(dst),plt.title("Warped Image")
#plt.show()
def trim(frame):
    #crop topA
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output.jpg",trim(dst))
