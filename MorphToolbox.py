# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:22:26 2024

@author: ramendra
"""

# Imprt necessary modules
from numba import jit
import numpy as np

# %% For greyscale erosion operation
# Simple loop operation with numba jit decorator
@jit(nopython = True)
def grey_erosion(x, SE):
    """        
    Return an eroded grey scale image after a single iteration of erosion by 
    the structuring element SE.

    Parameters:
            x (numpy 2D array): Input grey scale image in form of a numpy array
            SE (numpy 2D array): Structuring element for the morphological operation

    Returns:
            eroded_x (numpy 2D array): eroded array. Protrusion gets eroded.
    
    """
    half_width = int(SE.shape[0] / 2)
    x1 = np.max(x) * np.ones((x.shape[0] + 2 * half_width, 
                              x.shape[1] + 2 * half_width))
    x1[half_width : -half_width, half_width : -half_width] = x
    # x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))
    eroded_x = x1.copy()
    for i in range(half_width, x1.shape[0] - half_width):
        for j in range(half_width, x1.shape[1] - half_width):
            sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
            prod = sub_array * SE
            # eroded_x[i, j] = np.min(prod)
            eroded_x[i, j] = np.min(prod.ravel()[SE.ravel() != 0])
            
    eroded_x = eroded_x[half_width : -half_width, half_width : -half_width]
    return eroded_x

# %% For greyscale dilation operation
@jit(nopython = True)
def grey_dilation(x, SE):
    """        
    Return a dilated grey scale image after a single iteration of dilation by 
    the structuring element SE.

    Parameters:
            x (numpy 2D array): Input grey scale image in form of a numpy array
            SE (numpy 2D array): Structuring element for the morphological operation

    Returns:
            dilated_x (numpy 2D array): dilated array. Intrusions gets filled up.
    
    """
    half_width = int(SE.shape[0] / 2)
    x1 = np.min(x) * np.ones((x.shape[0] + 2 * half_width, 
                              x.shape[1] + 2 * half_width))
    x1[half_width : -half_width, half_width : -half_width] = x
    # x1 = np.pad(x, pad_width = half_width, mode='constant', constant_values = np.max(x))
    dilated_x = x1.copy()
    for i in range(half_width, x1.shape[0] - half_width):
        for j in range(half_width, x1.shape[1] - half_width):
            sub_array = x1[i - half_width : i + half_width + 1, j - half_width : j + half_width + 1]
            prod = sub_array * SE
            dilated_x[i, j] = np.max(prod.ravel()[SE.ravel() != 0])
            
    dilated_x = dilated_x[half_width : -half_width, half_width : -half_width]
    return dilated_x

# %% For greyscale opening operation
@jit(nopython = True)
def grey_opening(x, SE):
    """        
    Return a opened grey scale image after a single iteration of opening operation by 
    the structuring element SE. Only removes the protrusion smaller in size than
    the SE.

    Parameters:
            x (numpy 2D array): Input grey scale image in form of a numpy array
            SE (numpy 2D array): Structuring element for the morphological operation

    Returns:
            opened_x (numpy 2D array): opened array.
    
    """
    eroded_x = grey_erosion(x, SE)
    opened_x = grey_dilation(eroded_x, SE)
    return opened_x

# %% For greyscale closing operation
@jit(nopython = True)
def grey_closing(x, SE):
    """        
    Return a closed grey scale image after a single iteration of closing operation by 
    the structuring element SE. Only fills up the protrusion intrusion in size than
    the SE.

    Parameters:
            x (numpy 2D array): Input grey scale image in form of a numpy array
            SE (numpy 2D array): Structuring element for the morphological operation

    Returns:
            closed_x (numpy 2D array): closed array.
    
    """
    dilated_x = grey_dilation(x, SE)
    closed_x = grey_erosion(dilated_x, SE)
    return closed_x

