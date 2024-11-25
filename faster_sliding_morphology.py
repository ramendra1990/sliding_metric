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

# %% trying numba jit for checking processing speed
from numba import jit
import numpy as np
import timeit

x = np.arange(10000).reshape(100, 100)
@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

def just_go(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

@jit(nopython=True)
def go_fast(x, fx, fy, hx, hy): # Function is compiled and runs in machine code
    # import numpy as np
    num_frames_x = 1 + int((x.shape[1] - fx) / hx)
    num_frames_y = 1 + int((x.shape[0] - fy) / hy)
    min_x_array = np.zeros((num_frames_y, num_frames_x))
    for i in range(min_x_array.shape[0]):
        for j in range(min_x_array.shape[1]):
            ts = hy * i
            bs = (hy * i) + fy
            ls = hx * j
            rs = (hx * j) + fx
            sub_x = x[ts : bs, ls : rs]
            min_x_array[i, j] = np.min(sub_x)
            
    return min_x_array

def just_go(x, fx, fy, hx, hy):
    # import numpy as np
    num_frames_x = 1 + int((x.shape[1] - fx) / hx)
    num_frames_y = 1 + int((x.shape[0] - fy) / hy)
    min_x_array = np.zeros((num_frames_y, num_frames_x))
    for i in range(min_x_array.shape[0]):
        for j in range(min_x_array.shape[1]):
            ts = hy * i
            bs = (hy * i) + fy
            ls = hx * j
            rs = (hx * j) + fx
            sub_x = x[ts : bs, ls : rs]
            min_x_array[i, j] = np.min(sub_x)
            
    return min_x_array

# @jit(nopython=True). skimage can't be incorporated alongwith @jit decorator
def go_faster(x, fx, fy, hx, hy): # Function is compiled and runs in machine code
    from skimage.util import view_as_windows
    x_framed = view_as_windows(x, window_shape = (fy, fx), 
                               step = (hy, hx))
    min_x_windows = np.zeros((x_framed.shape[0], x_framed.shape[1]))
    for i in range(x_framed.shape[0]):
        for j in range(x_framed.shape[1]):
            min_x_windows[i, j] = np.min(x_framed[i, j, :, :])
            
    return min_x_windows

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = timeit.default_timer()
go_fast(x, 41, 41, 1, 1)
end = timeit.default_timer()
print("Elapsed (with compilation) = %s" % (end - start))

# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = timeit.default_timer()
# go_fast(x, 41, 41, 1, 1)
# end = timeit.default_timer()
# print("Elapsed (after compilation) = %s" % (end - start))

# skimage windows
start = timeit.default_timer()
go_faster(x, 41, 41, 1, 1)
end = timeit.default_timer()
print("Elapsed (with compilation) = %s" % (end - start))

# Normal numpy calculation
start = timeit.default_timer()
just_go(x, 41, 41, 1, 1)
end = timeit.default_timer()
print("Elapsed (after compilation) = %s" % (end - start))

# %% Chcking with morphological operation
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from MorphToolbox import *
import timeit

# ****************---------------*************
## For simple erosion -----------------#######
np.random.seed(3)
x = np.random.randint(1, 10000, size=(100, 100))
footprint = disk(20)
# transformed = erosion(x, footprint)

# ---------------------------
# Plotting
# data = transformed
# import matplotlib.pyplot as plt
# plt.figure()
# from mpl_toolkits import mplot3d
# ax = plt.axes(projection ='3d')
# X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
# ax.plot_surface(X, Y, data, antialiased=False)
# ----------------------------------

def simple_erosion(x, footprint):
    half_width = int(footprint.shape[0] / 2)
    x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))
    eroded_x = x1.copy()
    for i in range(half_width, x1.shape[0] - half_width):
        for j in range(half_width, x1.shape[1] - half_width):
            sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
            prod = sub_array * footprint
            eroded_x[i, j] = np.min(prod[footprint != 0])
            
    eroded_x = eroded_x[half_width : -half_width, half_width : -half_width]
    return eroded_x

def simple_dilation(x, footprint):
    half_width = int(footprint.shape[0] / 2)
    x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.min(x))
    dilated_x = x1.copy()
    for i in range(half_width, x1.shape[0] - half_width):
        for j in range(half_width, x1.shape[1] - half_width):
            sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
            prod = sub_array * footprint
            dilated_x[i, j] = np.max(prod[footprint != 0])
            
    dilated_x = dilated_x[half_width : -half_width, half_width : -half_width]
    return dilated_x

def simple_opening(x, footprint):
    eroded_x = simple_erosion(x, footprint)
    opened_x = simple_dilation(eroded_x, footprint)
    return opened_x

def simple_closing(x, footprint):
    dilated_x = simple_dilation(x, footprint)
    closed_x = simple_erosion(dilated_x, footprint)
    return closed_x

# half_width = int(footprint.shape[0] / 2)

# # frame_length_x = 3
# # frame_length_y = 3
# # hop_length_x = 1
# # hop_length_y = 1


# x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))


# eroded_x = x1.copy()
# for i in range(half_width, x1.shape[0] - half_width):
#     for j in range(half_width, x1.shape[1] - half_width):
#         sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
#         prod = sub_array * footprint
#         eroded_x[i, j] = np.min(prod[footprint != 0])

# eroded_x = eroded_x[half_width : -half_width, half_width : -half_width]

# ------------------------------------------
# Speed test

@jit(nopython = True)
def flatten_array(arr):
    flat_arr = np.empty(arr.shape[0] * arr.shape[1], dtype=arr.dtype)
    idx = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            flat_arr[idx] = arr[i, j]
            idx += 1
    return flat_arr

@jit(nopython = True)
def flatten_array1(arr):
    return arr.ravel()

# For loop inside jit-nopython decorator

@jit(nopython = True)
def fast_erosion(x, footprint):
    half_width = int(footprint.shape[0] / 2)
    x1 = np.max(x) * np.ones((x.shape[0] + 2 * half_width, 
                              x.shape[1] + 2 * half_width))
    x1[half_width : -half_width, half_width : -half_width] = x
    # x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))
    eroded_x = x1.copy()
    for i in range(half_width, x1.shape[0] - half_width):
        for j in range(half_width, x1.shape[1] - half_width):
            sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
            prod = sub_array * footprint
            # eroded_x[i, j] = np.min(prod)
            eroded_x[i, j] = np.min(prod.ravel()[footprint.ravel() != 0])
            
    eroded_x = eroded_x[half_width : -half_width, half_width : -half_width]
    return eroded_x

# import numba as nb
# @jit(nopython = True)
# def fast_erosion1(x, footprint):
#     half_width = int(footprint.shape[0] / 2)
#     x1 = np.max(x) * np.ones((x.shape[0] + 2 * half_width, 
#                               x.shape[1] + 2 * half_width))
#     x1[half_width : -half_width, half_width : -half_width] = x
#     # x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))
#     eroded_x = x1.copy()
#     for i in range(half_width, x1.shape[0] - half_width):
#         for j in range(half_width, x1.shape[1] - half_width):
#             sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
#             prod = sub_array * footprint
#             indx = np.where(footprint != 0)
#             # result = np.empty((indx[0].size,), dtype = prod.dtype)
#             result = []
#             for k in nb.literal_unroll(range(indx[0].size)): 
#                 result.append(prod[indx[0][k], indx[1][k]])
#             # eroded_x[i, j] = np.min(prod)
#             eroded_x[i, j] = np.min(np.array(result))
            
#     eroded_x = eroded_x[half_width : -half_width, half_width : -half_width]
#     return eroded_x

# ex = fast_erosion(x, footprint)

# # Flattening comarison
# start = timeit.default_timer()
# x_flat = flatten_array(x)
# end = timeit.default_timer()
# print("Elapsed = %s" % (end - start))

# start = timeit.default_timer()
# x_flat = x.flatten()
# end = timeit.default_timer()
# print("Elapsed = %s" % (end - start))

# start = timeit.default_timer()
# x_flat = flatten_array1(x)
# end = timeit.default_timer()
# print("Elapsed = %s" % (end - start))

# start = timeit.default_timer()
# x_flat = x.ravel()
# end = timeit.default_timer()
# print("Elapsed = %s" % (end - start))

# Comparison of skimage filter vs for loop with numba jit decorator
start = timeit.default_timer()
transformed1 = closing(x, footprint)
end = timeit.default_timer()
print("Elapsed (skimage filter) = %s" % (end - start))

start = timeit.default_timer()
transformed2 = simple_closing(x, footprint)
end = timeit.default_timer()
print("Elapsed (simple for loop) = %s" % (end - start))

start = timeit.default_timer()
transformed3 = grey_closing(x, footprint)
end = timeit.default_timer()
print("Elapsed (for loop with jit decorator) = %s" % (end - start))

# start = timeit.default_timer()
# transformed3 = fast_erosion1(x, footprint)
# end = timeit.default_timer()
# print("Elapsed (for loop with jit decorator and numba indexing) = %s" % (end - start))

# %% Random check/trial
@jit(nopython=True)
def manual_pad(array, pad_width, constant_values):
    half_width = int(array.shape[0] / 2)
    # Pad the array manually
    padded_array = np.ones((array.shape[0] + 2 * pad_width, 
                             array.shape[1] + 2 * pad_width))
    padded_array[pad_width:-pad_width, pad_width:-pad_width] = array

# array = np.array([[1, 2], [3, 4]])
# padded_array = manual_pad(array, 1, 0)

# import numba as nb
# @jit(nopython=True)
# def ar_min(ar):
#     # return np.min(ar[np.where(ar != 0)])
#     indx = np.where(ar != 0)
#     result = np.empty((indx[0].size,), dtype=ar.dtype)
#     for i in nb.literal_unroll(range(indx[0].size)): 
#         result[i] = ar[indx[0][i], indx[1][i]]
#     return np.min(result)


# ar_min(prod)

















