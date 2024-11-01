# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:27:18 2024

@author: ramendra
"""

import numpy as np
from numpy.lib import stride_tricks
import time

# %% for 1D numpy array
x = np.arange(1000)
x = np.random.randn(1000,)
frame_length = 30
hop_length = 10
num_frames = 1 + int((len(x) - frame_length) / hop_length)

#-------- Compare two methods (for loop and numpy stride method) 

# 1. numpy stride mthod
start_time = time.time_ns()
row_stride = x.itemsize * hop_length
col_stride = x.itemsize
x_framed_strided = stride_tricks.as_strided(x, 
                                            shape = (int(num_frames), frame_length), 
                                            strides=(row_stride, col_stride))
min_x = np.min(x_framed_strided, axis = 1)
# print(f"start time is {start_time}")
# print(f"start time is {time.time_ns()}")
print(f"time taken by the stride method is {(time.time_ns() - start_time)/1e6} micro secs", 
      f"length of min x is {min_x.shape[0]}")

# 2. With traditional for loop
start_time = time.time_ns()
min_x_array = np.zeros((num_frames,))
for i in range(min_x_array.shape[0]):
    ls = hop_length * i
    rs = (hop_length * i) + frame_length - 1
    sub_x = x[ls : rs]
    min_x_array[i] = int(np.min(sub_x))    
# print(f"start time is {start_time}")
# print(f"start time is {time.time_ns()}")
print(f"time taken by the for loop method is {(time.time_ns() - start_time)/1e6} micro secs", 
      f"length of min x is {min_x.shape[0]}")

# %% For 2D numpy array
np.random.seed(3)
x = np.random.randint(1, 100, size=(10, 10))
frame_length_x = 5
frame_length_y = 5
hop_length_x = 5
hop_length_y = 5

num_frames_x = 1 + int((x.shape[1] - frame_length_x) / hop_length_x)
num_frames_y = 1 + int((x.shape[0] - frame_length_y) / hop_length_y)
num_frames = num_frames_x * num_frames_y

# row_stride = x.itemsize * hop_length
# col_stride = x.itemsize
# to know th info for strides
desired_shape = (int(num_frames), frame_length_y, frame_length_x)
z = np.zeros(desired_shape)

x_framed_strided1 = stride_tricks.as_strided(x, 
                                            shape = (int(num_frames), frame_length_y, frame_length_x), 
                                            strides = (200, 20, 40, 4))   



# %%
from skimage.util import view_as_windows
x_framed = view_as_windows(x, (5, 5), step=2)















