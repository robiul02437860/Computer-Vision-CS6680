# -*- coding: utf-8 -*-
"""
Personal Details:
    name: Md. Robiul Islam
    A-number: A02437860
    Email: robiul@ece.ruet.ac.bd or a02437860@aggies.usu.edu
    Assignment Number 02: Image Enhancement in the Spatial Domain
    
Created on Fri Sep 13 08:58:45 2024

@author: robiul
"""
    
#import necessary python packages
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

import warnings
# Ignore all warnings
warnings.filterwarnings('ignore') #to remove warning from console of chi-squared divide by 0

#read original images
foodIm = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 2/Food.jpg", 0)

"""Assignment 2 question 1"""
"""------------------------------\
    ----------------------------"""

def LinearScaling(inputIm, intensity_range):
    """
    LinearScaling trasfer the old image intesities to a new range
    Parameters
    ----------
    inputIm : (np.array2d)
        Original image.
    intensity_range : (new scaling range)


    Returns
    -------
    scaledIm : (np.array (2d))
        Transformed image.
    transFunc : (1d array)
        Transfer function mapping
        
    """
    
    # Input validation
    if not isinstance(inputIm, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    if inputIm.dtype != np.uint8:
        raise ValueError("Input image must have uint8 data type.")
    if len(intensity_range) != 2:
        raise ValueError("Range must be a two-element vector.")
        
    if not(isinstance(intensity_range[0], int) and isinstance(intensity_range[1], int)):
        raise ValueError("range contains non-integer values.")
    if intensity_range[0]<0 or intensity_range[1]<0:
        raise ValueError("range contains negative values.")
    if intensity_range[0] > intensity_range[1]:
        raise ValueError("range contains a larger first element than the second element")
    
    #old intensity range
    old_min = inputIm.min()
    old_max = inputIm.max()
    
    # New intensity range
    new_min, new_max = intensity_range[0], intensity_range[1]
    
    # Calculate slope
    m = (new_max - new_min) / (old_max - old_min)
    
    #calculate the intercept
    b = new_min-m*old_min
    
    scaledIm = (inputIm*m+b).astype('uint8')
    
    
    # Initialize the transformation function (for all grayscale intensities from 0 to 255)
    transFunc = np.array(range(old_min, old_max+1))*m+b   #y = mx+b
    transFunc = transFunc.astype('uint8')
    
    
    return scaledIm, transFunc


#design of four user input cases
#Case 1
min_intensity = 20.0
max_intensity = 200.3
scaledFoodIm, transFunc = LinearScaling(foodIm, [min_intensity, max_intensity])

#Case 2
min_intensity = -20
max_intensity = -200
scaledFoodIm, transFunc = LinearScaling(foodIm, [min_intensity, max_intensity])

#Case 3
min_intensity = 200
max_intensity = 20
scaledFoodIm, transFunc = LinearScaling(foodIm, [min_intensity, max_intensity])

#Case 4
min_intensity = 20
max_intensity = 200
scaledFoodIm, transFunc = LinearScaling(foodIm, [min_intensity, max_intensity])

#plotting transFunc
x = np.array(range(foodIm.min(), foodIm.max()+1))
y=transFunc
plt.title("Transfer Function Plot") 
plt.xlabel("Old image scale") 
plt.ylabel("Transformed Image Scale") 
plt.plot(x, y)
plt.show()




"""Assignment 2 question 2"""
"""------------------------------\
    ----------------------------"""

def CalHist(image):
    """
    Calculate the normalized histogram of a grayscale input image.
    
    Parameters:
    image (numpy.ndarray): 2D array representing the grayscale image.
    
    Returns:
    numpy.ndarray: Normalized histogram array.
    """
    
    # Check that the input image is a 2D numpy array
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Initialize histogram array for 256 possible intensity levels
    hist  = np.array([0.0]*256) #same as hist = np.zeros(256, dtype=float)
    
    # Compute the histogram
    for y in range(height):
        for x in range(width):
            intensity = image[y, x]
            hist[intensity] += 1
    
    # Normalize the histogram
    total_pixels = height * width
    normalized_hist = hist / total_pixels
    
    return normalized_hist


nH1 = CalHist(foodIm)
nH2 = CalHist(scaledFoodIm)


def chi_square_dist(x, y):
    """
    Calculate the Chi-square distance between two normalized histograms.
    
    Parameters:
    x (numpy.ndarray): First histogram (normalized).
    y (numpy.ndarray): Second histogram (normalized).
    
    Returns:
    float: The Chi-square distance between the two histograms.
    """
    
    # Check that both histograms have the same size
    if x.shape != y.shape:
        raise ValueError("Histograms must have the same size.")
    
    # Compute the Chi-square distance
    # To avoid division by zero, we need to handle cases where x_i + y_i is zero
    # Create an array for the denominator
    denominator = x + y
    
    # Compute the numerator (xi - yi)^2
    numerator = (x - y) ** 2
    
    # To handle the case where denominator is zero, we'll use np.where to avoid division by zero
    chi_square = 0.5 * np.sum(np.where(denominator == 0, 0, numerator / denominator))
    
    return chi_square


def hist_intersection(x, y):
    """
    Calculate the histogram intersection between two normalized histograms.
    
    Parameters:
    x (numpy.ndarray): First histogram (normalized).
    y (numpy.ndarray): Second histogram (normalized).
    
    Returns:
    float: The histogram intersection between the two histograms.
    """
    
    # Check that both histograms have the same size
    if x.shape != y.shape:
        raise ValueError("Histograms must have the same size.")
    
    # Compute the histogram intersection
    intersection = np.sum(np.minimum(x, y))
    
    return intersection



# Display the normalized hists side by side
fig, ax = plt.subplots(1, 2)

# Plot the first histogram
ax[0].bar(range(len(nH1)), nH1, color='blue', alpha=0.7)
ax[0].set_title('Histogram (foodIm)')
ax[0].set_xlabel('Intensity Value')
ax[0].set_ylabel('Count')

# Plot the second histogram
ax[1].bar(range(len(nH2)), nH2, color='green', alpha=0.7)
ax[1].set_title('Histogram (scaledfoodIm)')
ax[1].set_xlabel('Intensity Value')
ax[1].set_ylabel('Count')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()


print("chi-squared distance is ", chi_square_dist(nH1, nH2))
print("Histogram intersection is ", hist_intersection(nH1, nH2))
print("\nInterpretation---> A high Chi-Squared distance means that there are large discrepancies between corresponding bins of the two hitograms\
      and a comparative low Histogram Intersection value means that there is little overlap between the two histograms.i. e. \
          two histograms being compared have significant differences")



"""Assignment 2 question 3"""
"""------------------------------\
    ----------------------------"""

def HistEqualization(inputIm):
    """
    Perform histogram equalization on a grayscale image.
    
    Parameters:
    inputIm (numpy.ndarray): Grayscale input image.
    
    Returns:
    enhancedIm (numpy.ndarray): Histogram equalized image.
    transFunc (numpy.ndarray): Transformation function (mapping of original intensities to new intensities).
    """
    
    if inputIm.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    
    # Step 1: Compute the histogram h(k)
    hist = np.zeros((256,), dtype=int)
    for intensity in range(256):
            # Use np.where() directly on the 2D image to count occurrences of each intensity
            hist[intensity] = np.sum(np.where(inputIm == intensity, 1, 0))
    
    # Step 2: Compute the cumulative normalized histogram cdf T(k)
    normalized_hist = hist/(inputIm.shape[0]*inputIm.shape[1])
    cdf_normalized = np.zeros_like(normalized_hist)
    running_total = 0
    for i in range(len(normalized_hist)):
        running_total+=normalized_hist[i]
        cdf_normalized[i] = running_total
    
    # Step 3: Compute the transformed intensity by: gk = (L-1) * T(k)
    transFunc = np.round(cdf_normalized*255).astype(np.uint8)
    enhancedIm = transFunc[inputIm]
        
    # Step 4: Map the old intensities to new intensities
    enhancedIm = transFunc[inputIm]
    
    return enhancedIm, transFunc

start = time.time()
# Perform histogram equalization
equalizedFoodIm, transFunc = HistEqualization(foodIm)
end = time.time()

# Calculate the runtime
runtime = end - start
print(f"Runtime of HistEqualization function: {runtime:.6f} seconds")




"""Assignment 2 question 4"""
"""------------------------------\
    ----------------------------"""

start = time.time()
# Perform histogram equalization using Built-in Function which does not return transfer function

equalized_image = cv2.equalizeHist(foodIm)
end = time.time()

# Calculate the runtime
runtime = end - start
print(f"Runtime of HistEqualization function: {runtime:.6f} seconds")



"""Assignment 2 question 5"""
"""------------------------------\
    ----------------------------"""
def BBHE(inputIm):
    inputIm = foodIm
    X = inputIm.flatten() #convert the 2d image array to a 1d array
    X_mean = X.mean() #input image mean
    
    #divide the input image into 2 subimages based on image mean
    X_L = X[X<=X_mean]
    X_U = X[X>X_mean]
    
    
    #Compute the histogram h(k) for X_L and X_U seperately
    Hist_L = np.zeros((round(X_mean)+1), dtype=int)
    Hist_U = np.zeros((255-round(X_mean)), dtype=int)
    
    #Histogram calculation for lower subimage
    for intensity in range(0, int(round(X_mean))+1): 
        Hist_L[intensity] = np.sum(np.where(X_L == intensity, 1, 0))
     #Histogram calculation for upper subimage
    for intensity in range(int(round(X_mean))+1, 256):
            Hist_U[intensity-round(X_mean)-1] = np.sum(np.where(X_U == intensity, 1, 0))
    
    
    #Compute the cumulative normalized histogram cdf T(k) of lower subimage
    normalized_hist_L = Hist_L/(X_L.shape[0])
    cdfL_normalized = np.zeros_like(normalized_hist_L)
    running_total = 0
    for i in range(len(normalized_hist_L)):
        running_total+=normalized_hist_L[i]
        cdfL_normalized[i] = running_total
    
    
    #Compute the cumulative normalized histogram cdf T(k) of upper subimage
    normalized_hist_U = Hist_U/(X_U.shape[0])
    cdfU_normalized = np.zeros_like(normalized_hist_U)
    running_total = 0
    for i in range(len(normalized_hist_U)):
        running_total+=normalized_hist_U[i]
        cdfU_normalized[i] = running_total
    
    
    
    #tranform function as whole
    transform_function = np.zeros(256, dtype=np.uint8)
    transform_function[:int(round(X_mean))+1] = (cdfL_normalized*int(round(X_mean))).astype(np.uint8)
    transform_function[int(round(X_mean))+1:] = (cdfU_normalized *(255-int(round(X_mean)+1)) + round(X_mean)+1).astype(np.uint8)
    
    # transfer the input image to enhanced image using the transformation function
    enhancedIm = transform_function[inputIm]

    
    return enhancedIm, transform_function


start = time.time()
# Perform histogram equalization using BBHE function
BBHEFoodIm, transFunc_ = BBHE(foodIm)
end = time.time()

# Calculate the runtime
runtime = end - start
print(f"Runtime of BBHE: {runtime:.6f} seconds")



#ploting enhanced images produced from 3, 4 and 5

fig, axes = plt.subplots(1, 3, figsize=(30, 30))

# 1st image from self implementation problem 3
axes[0].imshow(equalizedFoodIm, cmap="gray")
axes[0].set_title('equalizedFoodIm(Implementation)')

# 2nd image from built in function problem 4
axes[1].imshow(equalized_image, cmap="gray")
axes[1].set_title('equalized_image (Built-in)')

# 3rd image from BBHE method problem 5
axes[2].imshow(BBHEFoodIm, cmap="gray")
axes[2].set_title('BBHEFoodIm')

plt.show()



#plot the transform function (Buil-in function does not return TransFunc in this case)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

x = range(0, 256)
# Plot the first transformation function
axes[0].plot(x, transFunc, color='blue', marker='o')
axes[0].set_title("Transformation Function 1")
axes[0].set_xlabel("Original Intensity")
axes[0].set_ylabel("Transformed Intensity")
axes[0].grid(True)

# Plot the second transformation function
axes[1].plot(x, transFunc_, color='green', marker='o')
axes[1].set_title("Transformation Function 2")
axes[1].set_xlabel("Original Intensity")
axes[1].set_ylabel("Transformed Intensity")
axes[1].grid(True)
# Adjust layout
plt.tight_layout()
# Display the plots
plt.show()


#implemendation of Peak Signal to Noise Ratio (PSNR)

def compute_mse(image1, image2):
    """Compute Mean Squared Error between two images."""
    return np.mean((image1 - image2) ** 2)

def compute_psnr(original_image, transformed_image):
    """Compute PSNR between original image and transformed image."""
    mse = compute_mse(original_image, transformed_image)
    if mse == 0:
        return float('inf')  # If MSE is 0, PSNR is infinite
    max_pixel_value = 255.0  
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)      # R =255 in case of image
    return psnr


# Compute and display PSNR for both cases
psnr_equalized = compute_psnr(foodIm, equalizedFoodIm)
psnr_bbhe = compute_psnr(foodIm, BBHEFoodIm)

print(f"PSNR between original and equalized image: {psnr_equalized:.2f} dB")
print(f"PSNR between original and BBHE image: {psnr_bbhe:.2f} dB")
