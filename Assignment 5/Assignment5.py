# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 01:58:57 2024
Personal Details:
    name: Md. Robiul Islam
    A-number: A02437860
    Email: robiul@ece.ruet.ac.bd or a02437860@aggies.usu.edu
    Assignment Number 05: Morphological Operations

@author: robiul
"""
#import necessary python packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology



"""Problem I: Problem Solving Using Morphological Operations"""
"""Question 1"""
"""------------------------------\
    ----------------------------"""
    
# Load the image and convert it to grayscale
Wirebond = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 5/Wirebond.tif", 0)

#Converting it to 0 or 1 instead of 0 or 255
Wirebond[Wirebond==255]=1

# Define structuring elements (kernels)
kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14))
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

# (b) Apply erosion with the structuring element 
desired_image1 = cv2.erode(Wirebond, kernel_1, iterations=1)

# (c) Apply erosion with the structuring element
desired_image2 = cv2.erode(Wirebond, kernel_2, iterations=1)

# (d) Apply erosion with the structuring element
desired_image3 = cv2.erode(Wirebond, kernel_3, iterations=1)


# Plot the results
plt.figure(figsize=(12, 8))

# First dilated image
plt.subplot(1, 3, 1)
plt.imshow(desired_image1, cmap='gray')
plt.title('(b) Desired Image 1')
plt.axis('off')

# Second dilated image
plt.subplot(1, 3, 2)
plt.imshow(desired_image2, cmap='gray')
plt.title('(c) Desired Image 2')
plt.axis('off')

# Closed image
plt.subplot(1, 3, 3)
plt.imshow(desired_image3, cmap='gray')
plt.title('(d) Desired Image 3')
plt.axis('off')

# Display the plots
plt.tight_layout()
plt.show()


"""Problem I: Problem Solving Using Morphological Operations"""
"""Question 2"""
"""------------------------------\
    ----------------------------"""
    

# Load the image and convert it to grayscale
Shapes = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 5/Shapes.tif", 0)

#Converting it to 0 or 1 instead of 0 or 255
Shapes[Shapes==255]=1

#Structuring Element
kernel_1 = np.ones(shape=(20, 20))

# (b) Apply erosion followed by dilation
desired_image1 = cv2.erode(Shapes, kernel_1, iterations=1)
desired_image1 = cv2.dilate(desired_image1, kernel_1, iterations=1)


# (c) Apply dilation followed by erosion
desired_image2 = cv2.dilate(Shapes, kernel_1, iterations=1)
desired_image2 = cv2.erode(desired_image2, kernel_1, iterations=1)

# (d) Apply dilation followed by erosion
desired_image3 = cv2.dilate(Shapes, kernel_1, iterations=1)
desired_image3 = cv2.erode(desired_image3, kernel_1, iterations=1)
desired_image3 = cv2.erode(desired_image3, kernel_1, iterations=1)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(desired_image1, cmap='gray')
plt.title('(f) Desired image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(desired_image2, cmap='gray')
plt.title('(g) Desired image 2')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(desired_image3, cmap='gray')
plt.title('(h) Desired image 3')
plt.axis('off')

# Display the plots
plt.tight_layout()
plt.show()



"""Problem I: Problem Solving Using Morphological Operations"""
"""Question 3"""
"""------------------------------\
    ----------------------------"""
    
from skimage import io, morphology, filters, measure

# Load the image
Dowels = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 5/Dowels.tif", 0)

# Convert the grayscale image to binary using Otsu's thresholding
threshold_value = filters.threshold_otsu(Dowels)

binary_image = (Dowels > threshold_value).astype('uint8')

# Define the disk structuring element
def disk(radius):
    return morphology.disk(radius)

# Apply open-close operation with radius 5
open_close_image = morphology.closing(morphology.opening(binary_image, disk(5)), disk(5))

# Apply close-open operation with radius 5
close_open_image = morphology.opening(morphology.closing(binary_image, disk(5)), disk(5))

# Display the results side-by-side as Figure 3
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6))
ax3[0].imshow(open_close_image, cmap='gray')
ax3[0].set_title('Open-Close Operation')
ax3[0].axis('off')
ax3[1].imshow(close_open_image, cmap='gray')
ax3[1].set_title('Close-Open Operation')
ax3[1].axis('off')
plt.show()



# Apply a series of open-close operations
open_close_series = [morphology.closing(morphology.opening(binary_image, disk(r)), disk(r)) for r in range(2, 6)]

# Apply a series of close-open operations
close_open_series = [morphology.opening(morphology.closing(binary_image, disk(r)), disk(r)) for r in range(2, 6)]

# Display the results side-by-side as Figure 4
fig4, ax4 = plt.subplots(2, 4, figsize=(16, 8))
for i, (oc_img, co_img) in enumerate(zip(open_close_series, close_open_series)):
    ax4[0, i].imshow(oc_img, cmap='gray')
    ax4[0, i].set_title(f'Open-Close Radius {i+2}')
    ax4[0, i].axis('off')
    ax4[1, i].imshow(co_img, cmap='gray')
    ax4[1, i].set_title(f'Close-Open Radius {i+2}')
    ax4[1, i].axis('off')
plt.show()

"""Problem II: Problem Solving Using Morphological Operations"""
"""Question 3"""
"""------------------------------\
    ----------------------------"""

# Load the binary image
image = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 5/SmallSquares.tif", 0)

#Converting it to 0 or 1 instead of 0 or 255
image[image==255] =1


hit_element = np.array([[0, 1, 0],
                        [0,  1, 1],
                        [0,  0, 0]], dtype='uint8')

hit_result = cv2.erode(image, hit_element)

miss_element = np.array([[1, 0, 0],
                        [1,  0, 0],
                        [1,  1, 1]], dtype='uint8')

miss_result = cv2.erode(~image, miss_element)


result = hit_result & miss_result

plt.figure()
plt.imshow(result, cmap='gray')
plt.title("Detected Pattern")
plt.axis('off')

plt.show()

# Output the number of foreground pixels that satisfy the condition
print("Number of foreground pixels that satisfy the conditions:", np.sum(result))


"""Problem III: Applications of Morphological Operations"""
"""Question 1"""
"""------------------------------\
    ----------------------------"""

def FindComponentLabels(im, se):
    # Initialize label image and label counter
    labelIm = np.zeros_like(im, dtype=int)
    label = 0
    
    # Iterate over all pixels in the binary image
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # If we find an unlabeled foreground pixel, start a new component
            if im[i, j] == 1 and labelIm[i, j] == 0:
                label += 1  # Increment label for a new component
                
                # Start with a seed point for the connected component
                component = np.zeros_like(im, dtype=bool)
                component[i, j] = True  # Initialize with the current point
                
                # Iteratively apply dilation to find the whole component
                while True:
                    # Perform dilation and intersect with the original image
                    next_component = morphology.binary_dilation(component, se) & im
                    if np.array_equal(next_component, component):
                        break  # Stop if no new pixels are added
                    component = next_component  # Update component
                
                # Label all pixels in the component
                labelIm[component] = label

    # Output the number of connected components
    num = label
    return labelIm, num

# Load the binary image
image = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 5/Ball.tif", 0)

#convert to binary image
binary_image = image>0

# Define a structuring element for connectivity ( 8-connectivity)
structuring_element = morphology.square(3)


# Call FindComponentLabels to label the connected components
labelIm, num = FindComponentLabels(binary_image, structuring_element)


# Print the number of connected components
print("Total number of connected particles:", num)

# Display the labeled image using color intensities
plt.figure(figsize=(8, 8))
plt.imshow(labelIm, cmap='nipy_spectral')
plt.colorbar(label="Component Labels")
plt.title(f"Labeled Connected Components (Total: {num})")
plt.axis('off')
plt.show()



"""Problem III: Applications of Morphological Operations"""
"""Question 2"""
"""------------------------------\
    ----------------------------"""

from scipy.ndimage import label

# scipy's label function to find connected components
labeled_image, num_features = label(binary_image, structure=np.ones((3, 3)))

# Display the labeled connected components with a colormap
plt.figure(figsize=(8, 8))
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.colorbar(label="Component Labels")
plt.title(f"Labeled Connected Components with Built-in Function (Total: {num_features})")
plt.axis('off')
plt.show()

# Print the total number of connected particles found
print("Total number of connected particles:", num_features)



"""Problem III: Applications of Morphological Operations"""
"""Question 3"""
"""------------------------------\
    ----------------------------"""

# Create an empty array A to store only the border-connected particles
A = np.zeros_like(binary_image, dtype=int)

# Define a 3x3 structuring element for 8-connectivity
se = morphology.square(3)

# Label all connected components in the image
labelIm, num = FindComponentLabels(binary_image, se)

# Identify border-connected components
border_labels = set()

# Check the top and bottom rows
for j in range(binary_image.shape[1]):
    if labelIm[0, j] > 0:
        border_labels.add(labelIm[0, j])
    if labelIm[-1, j] > 0:
        border_labels.add(labelIm[-1, j])

# Check the left and right columns
for i in range(binary_image.shape[0]):
    if labelIm[i, 0] > 0:
        border_labels.add(labelIm[i, 0])
    if labelIm[i, -1] > 0:
        border_labels.add(labelIm[i, -1])

# Create image A containing only border-connected components
for label in border_labels:
    A[labelIm == label] = 1
        
# Count the number of border-connected components
border_count = len(border_labels)

# Display the original and result images side-by-side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(A, cmap='gray')
plt.title("Border-Connected Particles")
plt.axis('off')

plt.show()

# Print the number of border-connected particles
print("Number of connected particles residing on the border:", border_count)



"""Problem III: Applications of Morphological Operations"""
"""Question 4"""
"""------------------------------\
    ----------------------------"""

# Define a 3x3 structuring element for 8-connectivity
se = morphology.square(3)

# Label all connected components in the image
labelIm, num = FindComponentLabels(binary_image, se)

# Identify border-connected particles by label
border_labels = set()

# Check the top and bottom rows
for j in range(binary_image.shape[1]):
    if labelIm[0, j] > 0:
        border_labels.add(labelIm[0, j])
    if labelIm[-1, j] > 0:
        border_labels.add(labelIm[-1, j])

# Check the left and right columns
for i in range(binary_image.shape[0]):
    if labelIm[i, 0] > 0:
        border_labels.add(labelIm[i, 0])
    if labelIm[i, -1] > 0:
        border_labels.add(labelIm[i, -1])

# Calculate the area of each component
regions = measure.regionprops(labelIm)
particle_areas = [region.area for region in regions if region.label not in border_labels]

# Estimate size of individual particles by finding the most common area
from scipy.stats import mode
individual_particle_area = mode(particle_areas)[0]


# Create empty images for overlapping and individual particles
B = np.zeros_like(binary_image, dtype=int)
C = np.zeros_like(binary_image, dtype=int)

# Classify each particle based on its area
overlapping_count = 0
individual_count = 0


for region in regions:
    if region.label in border_labels:
        continue  # Skip border particles

    if region.area > individual_particle_area*1.5:  # Adjust threshold as needed
        # Particle is larger than typical, classify as overlapping
        B[labelIm == region.label] = 1
        overlapping_count += 1
    else:
        # Particle matches typical size, classify as individual
        C[labelIm == region.label] = 1
        individual_count += 1
        

# Display the original image, and the classified images side-by-side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(B, cmap='gray')
plt.title("Overlapping Particles (Image B)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(C, cmap='gray')
plt.title("Individual Particles (Image C)")
plt.axis('off')

plt.show()

# Print counts of overlapping and individual particles not on the border
print("Number of overlapping particles not on the border:", overlapping_count)
print("Number of individual particles not on the border:", individual_count)


"""Problem III: Applications of Morphological Operations"""
"""Question 5"""
"""------------------------------\
    ----------------------------"""
    
# Define a 3x3 structuring element for 8-connectivity
se = morphology.square(3)

# Label all connected components in the image
labelIm, num = FindComponentLabels(binary_image, se)

# Identify border-connected particles by label
border_labels = set()

# Check the top and bottom rows
for j in range(binary_image.shape[1]):
    if labelIm[0, j] > 0:
        border_labels.add(labelIm[0, j])
    if labelIm[-1, j] > 0:
        border_labels.add(labelIm[-1, j])

# Check the left and right columns
for i in range(binary_image.shape[0]):
    if labelIm[i, 0] > 0:
        border_labels.add(labelIm[i, 0])
    if labelIm[i, -1] > 0:
        border_labels.add(labelIm[i, -1])

# Calculate the area of each component
regions = measure.regionprops(labelIm)
particle_areas = [region.area for region in regions if region.label not in border_labels]

# Estimate size of individual particles by finding the most common area
from scipy.stats import mode
individual_particle_area = mode(particle_areas)[0]

# Create empty images for partial overlapping and individual particles on border
D = np.zeros_like(binary_image, dtype=int)
E = np.zeros_like(binary_image, dtype=int)

# Classify each border-connected particle based on its area
partial_overlapping_count = 0
partial_individual_count = 0


for region in regions:
    if region.label in border_labels:
        if region.area > individual_particle_area*1.5:  # Adjust threshold as needed
            # Particle is larger than typical, classify as partial overlapping
            D[labelIm == region.label] = 1
            partial_overlapping_count += 1
        else:
            # Particle matches typical size, classify as individual
            E[labelIm == region.label] = 1
            partial_individual_count += 1


# Display the original image, and the classified images side-by-side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(D, cmap='gray')
plt.title("Partial Overlapping Particles on Border (Image D)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(E, cmap='gray')
plt.title("Partial Individual Particles on Border (Image E)")
plt.axis('off')

plt.show()

# Print counts of partial overlapping and individual particles on the border
print("Number of visually partial overlapping particles on the border:", partial_overlapping_count)
print("Number of visually partial individual particles on the border:", partial_individual_count)

