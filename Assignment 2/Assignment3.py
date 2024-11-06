# -*- coding: utf-8 -*-
"""
Personal Details:
    name: Md. Robiul Islam
    A-number: A02437860
    Email: robiul@ece.ruet.ac.bd or a02437860@aggies.usu.edu
    Assignment Number 03: Filter Techniques for Image Enhancement, Edge Detection, and Noise Removal
    
Created on Sat Sep 28 15:34:42 2024

@author: robiul
"""
#import necessary python packages
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt

#read circuit image
Circuit = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Circuit.jpg", 0)

"""Assignment 3 question 1"""
"""------------------------------\
    ----------------------------"""

def AverageFiltering(image, filter):
    # Check if the filter is a square matrix
    if filter.ndim != 2 or filter.shape[0] != filter.shape[1]:
        raise ValueError("Filter must be a square matrix.")
    
    # Check if the filter has odd dimensions
    filter_size = filter.shape[0]
    if filter_size % 2 == 0:
        raise ValueError("Filter must have odd dimensions.")
    
    # Check if all elements in the filter are positive
    for row in filter:
        for element in row:
            if element < 0:
                raise ValueError("All elements in the filter must be positive.")
    
    # Check if the sum of all elements in the filter is 1
    total_sum = 0
    for row in filter:
        for element in row:
            total_sum += element
    if abs(total_sum - 1) > 1e-6:
        raise ValueError("The sum of all elements in the filter must be 1.")

    # Check if the filter is symmetric around the center
    center = filter_size // 2
    for i in range(filter_size):
        for j in range(filter_size):
            if filter[i][j] != filter[filter_size - 1 - i][filter_size - 1 - j]:
                raise ValueError("The filter must be symmetric around the center.")
    
    
    # Get image dimensions
    image_height, image_width = image.shape
    
    # Initialize the output image
    #output_image = np.zeros_like(image, dtype=np.uint8)
    filtered_image = np.zeros_like(image)
    
    # Calculate the padding size
    pad = filter_size // 2
    
    # Padding the image to handle borders
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    
 
    # Perform convolution
    for i in range(pad, image_height + pad):
        for j in range(pad, image_width + pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            filtered_image[i-pad, j-pad] = np.sum(region * filter)

    return filtered_image

#Test case for elements in filter are positive
filter = np.array([[1/9, 1/9, 1/9],
                    [1/9, 1/9, -1/9],
                    [1/9, 1/9, 1/9]])

AverageFiltering(Circuit, filter)

#Test case for the sum of all elements is 1
filter = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

AverageFiltering(Circuit, filter)

#Test case for elements in filter are symmetric around center
filter = (1/23)*np.array([[1, 2, 1],
                           [3, 4, 6],
                           [2, 1, 3]])

AverageFiltering(Circuit, filter)

#call the AverageFiltering function using a standard 5×5 averaging filter
st_filter = (1/25)*np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]])

st_filter_output = AverageFiltering(Circuit, st_filter)

weighted_filter = (1/16)*np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])

wt_filter_output = AverageFiltering(Circuit, weighted_filter)

#ploting enhanced images produced

fig, axes = plt.subplots(1, 3, figsize=(30, 30))

# 1st image from self implementation problem 3
axes[0].imshow(Circuit, cmap="gray")
axes[0].set_title('Original Image')

# 2nd image from built in function problem 4
axes[1].imshow(st_filter_output, cmap="gray")
axes[1].set_title('Standard 5x5 filter processed')

# 3rd image from BBHE method problem 5
axes[2].imshow(wt_filter_output, cmap="gray")
axes[2].set_title('Weighted 3x3 filter processed')
plt.show()



"""Assignment 3 question 2"""
"""------------------------------\
    ----------------------------"""

def MedianFiltering(image, filter):
    # Check if the input image is a 2D array (grayscale image)
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale image).")
    
    size = len(filter)
    
    if size % 2 == 0:
        raise ValueError("Filter must have odd dimensions.")
    
    for row in filter:
        if len(row) != size:
            raise ValueError("Filter must be a square matrix.")
        for element in row:
            if element <= 0:
                raise ValueError("All elements in the filter must be positive integers.")
    
    # Get the filter size from the weights matrix
    filter_size = filter.shape[0]
    pad_size = filter_size // 2
    
    # Get the dimensions of the input image
    rows, cols = image.shape
    
    # Create a padded version of the image to handle borders
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    # Initialize the output image with the same shape and data type
    output_image = np.zeros((rows, cols), dtype=np.uint8)
    
    # Loop through each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood around the current pixel
            neighborhood = padded_image[i:i + filter_size, j:j + filter_size]
            
            # Flatten the neighborhood and the weights
            neighborhood_flat = neighborhood.flatten()
            weights_flat = filter.flatten()
            
            # Apply the weights by repeating each pixel value according to its weight
            weighted_pixels = np.repeat(neighborhood_flat, weights_flat)
            
            # Compute the median of the weighted pixel list
            median_value = np.median(weighted_pixels)
            
            # Set the median value to the output image
            output_image[i, j] = median_value
    
    return output_image


#Test case for elements in filter are positive
filter = np.array([[1, 1, -1],
                    [1, 1, 1],
                    [1, -1, 1]])

Median_output = MedianFiltering(Circuit, filter)

#call the MedianFiltering function using a standard 3x3 median filter
sd_med_filter = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

sd_median_output = MedianFiltering(Circuit, sd_med_filter)

#call the MedianFiltering function using a weighted 3x3 median filter
M = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]])

weighted_median_output = MedianFiltering(Circuit, M)


#Displaying enhanced images produced by median filtering

fig, axes = plt.subplots(1, 3, figsize=(30, 30))

# 1st image from self implementation problem 3
axes[0].imshow(Circuit, cmap="gray")
axes[0].set_title('Original Image')

# 2nd image from built in function problem 4
axes[1].imshow(sd_median_output, cmap="gray")
axes[1].set_title('Standard 3x3 median filter processed')

# 3rd image from BBHE method problem 5
axes[2].imshow(weighted_median_output, cmap="gray")
axes[2].set_title('Weighted 3x3 median filter processed')
plt.show()


"""Assignment 3 question 3"""
"""------------------------------\
    ----------------------------"""

# Load the grayscale image (e.g., the Moon image)
Original_Image = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Moon.jpg", 0)
# A strong 3x3 Laplacian mask (use one of the two strong Laplacian masks)

laplacian_mask = np.array([[1,  1,  1],
                           [1,  -8,  1],
                           [1,  1,  1]])

# Apply the Laplacian filter to the Moon using built-in function
Filtered_Image = cv2.filter2D(Original_Image, -1, laplacian_mask)
    
# Enhanced Image = Original Image - Filtered Image
Enhanced_Image = Original_Image-Filtered_Image


# Display the original and enhanced images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(30, 30))

# Original Moon Image
axes[0].imshow(Original_Image, cmap="gray")
axes[0].set_title('Original Image')

# Final Enhanced Image
axes[1].imshow(Enhanced_Image, cmap="gray")
axes[1].set_title('Enhanced Image')

plt.show()

"""Assignment 3 Problem II: Exercises on Edge Detectors in the Spatial Domain"""
"""------------------------------\
    ----------------------------"""
#summerizing the best strageties to find  threshold value
print("To determine the threshod for finding edge using sobel operator, we can use Otsu's method, adaptive thresolding,\
      entropy-based thresolding and percentile-based thresholding method.")
print("To determine the threshod for finding edge using canny edge detector, we can use Otsu's method (double method), \
      adaptive thresolding, entropy-based thresholding and percentile-based thresholding method.")
     
image = cv2.imread('C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Rice.jpg', cv2.IMREAD_GRAYSCALE)

#For automaticaaly finding the threshold value
def otsu_threshold(image):
    """Compute Otsu's threshold to separate foreground and background."""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    current_max, threshold = 0, 0
    sumB, wB, wF = 0, 0, 0
    sum1 = np.dot(np.arange(256), hist)

    for i in range(256):
        wB += hist[i]  # Weight Background
        wF = total_pixels - wB  # Weight Foreground
        if wB == 0 or wF == 0:
            continue
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        # Calculate Between Class Variance
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i
    return threshold


def FindEdgeInfo(image, num_bins):
    """Applies Sobel filters to detect edges and computes a histogram of edge orientations."""
    img_height, img_width = image.shape
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    # Apply Sobel filters to get gradients in Y direction Gx
    kernel_size = sobel_x.shape[0]  # Assuming square kernel
    pad = kernel_size // 2
    # Padding the image to handle borders
    padded_image = np.pad(image, pad, mode='reflect').astype('float')
    #Gx = np.zeros_like(image)
    Gx = np.zeros((img_height, img_width))
    # Perform convolution to get Gx
    for i in range(pad, img_height + pad):
        for j in range(pad, img_width + pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            Gx[i-pad, j-pad] = np.sum(region * sobel_x)

    #Gy = np.zeros_like(image)
    Gy = np.zeros((img_height, img_width))
    # Perform convolution to get Gx
    for i in range(pad, img_height + pad):
        for j in range(pad, img_width + pad):
            region = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
            Gy[i-pad, j-pad] = np.sum(region * sobel_y)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    threshold = otsu_threshold(image)
    #edges = gradient_magnitude > threshold
    important_edges = (gradient_magnitude >= threshold).astype(np.uint8)*255
    
    # Compute the gradient orientation in degrees
    orientations = np.degrees(np.arctan2(Gy, Gx))  # (-180 to 180 degrees)

    # Normalize the orientations to the range (0 to 360)
    orientations = np.mod(orientations + 360, 360)

    # Compute the histogram (count) of orientations 
    num_bins =30 
    bin_width = 360 / num_bins
    histogram = np.zeros(num_bins, dtype=np.int32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if important_edges[i, j] == 255:  # Only consider edges
                angle = orientations[i, j]
                bin_idx = int(angle // bin_width)
                histogram[bin_idx] += 1

    return important_edges, histogram



# Load the grayscale image (Rice image)
image = cv2.imread('C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Rice.jpg', cv2.IMREAD_GRAYSCALE)
# Call FindEdgeInfo with 30 bins for the edge orientation histogram

important_edges, edge_histogram = FindEdgeInfo(image, 30)


# Display the original image, edge-detected image, and the histogram
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 3, 2)
plt.imshow(important_edges, cmap='gray')
plt.title('Important Edges')
plt.axis('off')

# Edge orientation histogram
plt.subplot(1, 3, 3)
plt.bar(range(30), edge_histogram, width=0.8)
plt.title('30-bin Edge Orientation Histogram')
plt.xlabel('Bin')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


"""Assignment 3 Problem III: A Practical Problem"""
"""------------------------------\
    ----------------------------"""
def RemoveStripes(image):
    # Apply Sobel operator to calculate gradients
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
    # Calculate the angle of the gradient
    angles = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Convert to degrees
    
    # Normalize the angles to the range [0, 180]
    angles[angles < 0] += 180
    
    # Create a mask for horizontal edges
    # Horizontal edges are typically at angles close to 0 or 180 degrees
    # Define a threshold for horizontal angles
    horizontal_mask = (angles < 25) | (angles > 160)  # Threshold can be adjusted
    
    # Create a binary edge mask
    edges_binary = np.zeros_like(image, dtype=np.uint8)
    edges_binary[horizontal_mask] = 255  # Set horizontal edges to white
    
    # Optional: Invert the edges binary mask to show horizontal edges
    edges_binary = cv2.bitwise_not(edges_binary)
    
    # Inpaint to remove horizontal edges
    cleaned_image = cv2.inpaint(image, edges_binary, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned_image
#call removestripes() for the "Text.gif" image
image = Image.open("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Text.gif")
image = np.array(image)
cleaned_image = RemoveStripes(image)

#displaying original images and cleaned images
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(cleaned_image, cmap='gray')
plt.title('Important Edges')
plt.axis('off')

plt.tight_layout()
plt.show()

#call removestripes() for the "Text1.gif" image
image = Image.open("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 3/Text1.gif")
image = np.array(image)
cleaned_image = RemoveStripes(image)

#displaying original images and cleaned images
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.imshow(cleaned_image, cmap='gray')
plt.title('Important Edges')
plt.axis('off')

plt.tight_layout()
plt.show()

