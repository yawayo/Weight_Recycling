import numpy as np
from scipy.ndimage import rotate


weight_0 = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
print("weight_0", weight_0.shape)
print(weight_0)
print("----------------------------------------")
weight_90 = rotate(weight_0, 90, axes=(1,0,0,0))
print("weight_90", weight_90.shape)
print(weight_90)
print("----------------------------------------")
weight_180 = rotate(weight_90, 90, axes=(1,0,0,0))
print("weight_180", weight_180.shape)
print("----------------------------------------")
weight_270 = rotate(weight_180, 90, axes=(1,0,0,0))
print("weight_270", weight_270.shape)
print("----------------------------------------")
