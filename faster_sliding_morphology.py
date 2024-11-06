# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:27:18 2024

@author: ramendra
"""

import numpy as np
from numpy.lib import stride_tricks
from skimage.util import view_as_windows
import time

def time_in_picoseconds():
    return time.perf_counter_ns() * 1000

# %% for 1D numpy array
# x = np.arange(1000)
x = np.random.randint(1, 100, size = 1000000)
frame_length = 1000
hop_length = 200
num_frames = 1 + int((len(x) - frame_length) / hop_length)

#-------- Compare three methods (numpy stride method, for loop and skimage view_as_windows) 

# 1. numpy stride mthod
# start_time = time.time_ns()
start_time = time_in_picoseconds()
row_stride = x.itemsize * hop_length
col_stride = x.itemsize
x_framed_strided = stride_tricks.as_strided(x, 
                                            shape = (int(num_frames), frame_length), 
                                            strides=(row_stride, col_stride))
min_x = np.min(x_framed_strided, axis = 1)
# print(f"start time is {start_time}")
# print(f"start time is {time.time_ns()}")
print(f"time taken by the stride method is {(time_in_picoseconds() - start_time)/1e3} nano secs", 
      f"length of min x is {min_x.shape[0]}")

# 2. With traditional for loop
start_time = time.time_ns()
min_x_array = np.zeros((num_frames,))
for i in range(min_x_array.shape[0]):
    ls = hop_length * i
    rs = (hop_length * i) + frame_length
    sub_x = x[ls : rs]
    min_x_array[i] = int(np.min(sub_x))    
# print(f"start time is {start_time}")
# print(f"start time is {time.time_ns()}")
print(f"time taken by the for loop method is {(time.time_ns() - start_time)/1e1} micro secs", 
      f"length of min x is {min_x.shape[0]}")

# 3. Skimage view as windows method
start_time = time_in_picoseconds()
x_framed = view_as_windows(x, window_shape = frame_length, step = hop_length)
min_x = np.min(x_framed, axis = 1)
print(f"time taken by the skimage method is {(time_in_picoseconds() - start_time)/1e3} nano secs", 
      f"length of min x is {min_x.shape[0]}")

# %% For 2D numpy array
np.random.seed(3)
x = np.random.randint(1, 100, size=(4000, 4000))
frame_length_x = 100
frame_length_y = 100
hop_length_x = 50
hop_length_y = 50

# 1. Skimage view as windows method
start_time = time_in_picoseconds()
x_framed = view_as_windows(x, window_shape = (frame_length_y, frame_length_x), 
                           step = (hop_length_y, hop_length_x))
min_x_windows = np.zeros((x_framed.shape[0], x_framed.shape[1]))
for i in range(x_framed.shape[0]):
    for j in range(x_framed.shape[1]):
        min_x_windows[i, j] = np.min(x_framed[i, j, :, :])
        
print(f"time taken by the skimage method is {(time_in_picoseconds() - start_time)/1e3} nano secs", 
      f"shape of min x is {min_x_windows.shape}")

# 2. With traditional for loop
start_time = time_in_picoseconds()
num_frames_x = 1 + int((x.shape[1] - frame_length_x) / hop_length_x)
num_frames_y = 1 + int((x.shape[0] - frame_length_y) / hop_length_y)
min_x_array = np.zeros((num_frames_y, num_frames_x))
for i in range(min_x_array.shape[0]):
    for j in range(min_x_array.shape[1]):
        ts = hop_length_y * i
        bs = (hop_length_y * i) + frame_length_y
        ls = hop_length_x * j
        rs = (hop_length_x * j) + frame_length_x
        sub_x = x[ts : bs, ls : rs]
        min_x_array[i, j] = np.min(sub_x)    

print(f"time taken by the loop method is {(time_in_picoseconds() - start_time)/1e3} nano secs", 
      f"shape of min x is {min_x_array.shape}")


# num_frames_x = 1 + int((x.shape[1] - frame_length_x) / hop_length_x)
# num_frames_y = 1 + int((x.shape[0] - frame_length_y) / hop_length_y)
# num_frames = num_frames_x * num_frames_y

# # row_stride = x.itemsize * hop_length
# # col_stride = x.itemsize
# # to know th info for strides
# desired_shape = (int(num_frames), frame_length_y, frame_length_x)
# z = np.zeros(desired_shape)

# x_framed_strided = stride_tricks.as_strided(x, 
#                                             shape = (int(num_frames), frame_length_y, frame_length_x), 
#                                             strides = z.strides)   

# %%





















