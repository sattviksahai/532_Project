import numpy as np
import cv2

dpm = np.load("depthmaps/depth_0.npy")
dpm -= np.min(dpm)
dpm = (dpm*255)/ np.max(dpm)
# print(dpm)
dpm = np.array(dpm, dtype=np.uint8)
# print(np.min(dpm), np.max(dpm), dpm.shape)
dpm = cv2.resize(dpm, None, fx=0.3, fy=0.3)

dpm1 = np.load("depthmaps/depth_1.npy")
dpm1 -= np.min(dpm1)
dpm1 = (dpm1*255)/ np.max(dpm1)
# print(dpm)
dpm1 = np.array(dpm1, dtype=np.uint8)
# print(np.min(dpm), np.max(dpm), dpm.shape)
dpm1 = cv2.resize(dpm1, None, fx=0.3, fy=0.3)

cv2.imwrite("depth1.jpg", dpm1)
cv2.imshow("depth1", dpm1)
cv2.imshow("depth", dpm)
cv2.waitKey(0)
cv2.destroyAllWindows()