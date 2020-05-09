import numpy as np
import cv2

dpm = np.load("depth.npy")
dpm -= np.min(dpm)
dpm = (dpm*255)/ np.max(dpm)
print(dpm)
dpm = np.array(dpm, dtype=np.uint8)
print(np.min(dpm), np.max(dpm), dpm.shape)
dpm = cv2.resize(dpm, None, fx=0.3, fy=0.3)
cv2.imshow("depth", dpm)
cv2.waitKey(0)
cv2.destroyAllWindows()