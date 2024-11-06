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
import pywt
import matplotlib.pyplot as plt


"""Problem 1: Color Image Processing"""
"""------------------------------\
    ----------------------------"""
    
"""Question 1"""

# Load the image
image = cv2.imread('C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/ball.bmp')

# Step 2: Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Extract the Hue channel
hue_channel = hsv_image[:, :, 0]

# Display intermediate results - Original image and Hue channel
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# HSV Image (Hue Channel)
plt.subplot(1, 3, 2)
plt.title("Hue Channel")
plt.imshow(hue_channel, cmap='hsv')
plt.axis('off')

# Step 4: Apply thresholding to the Hue channel to separate the ball
# Define hue range for the orange color
lower_hue = 5
upper_hue = 25
mask = cv2.inRange(hue_channel, lower_hue, upper_hue)

# Display the thresholded Hue mask
plt.subplot(1, 3, 3)
plt.title("Thresholded Hue Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# Step 5: Find contours to locate the ball
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Identify the largest contour and compute its centroid
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    # Calculate centroid coordinates
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0  # Fallback if zero division occurs
    
    # Step 7: Draw a blue cross at the centroid on the original image
    marked_image = image.copy()
    cv2.drawMarker(marked_image, (cx, cy), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, 
                   markerSize=15, thickness=2)

    # Display the result - original image with centroid marked
    plt.figure(figsize=(5, 5))
    plt.title("Figure 2: Centroid Marked on Original Image")
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("No bright orange object detected in the image.")
    

"""Question 2"""


# Step 1: Load the image and convert it to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the original image to RGB for display with matplotlib
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Detect the ball's position as in the previous solution
# Extract the Hue channel and threshold for the orange color
hue_channel = hsv_image[:, :, 0]
lower_hue = 5
upper_hue = 25
ball_mask = cv2.inRange(hue_channel, lower_hue, upper_hue)

# Find contours to locate the ball
contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Step 3: Define a region around the ball for shadow detection
    region_size = 150  # Adjust size to control search area around the ball
    x_min = max(cx - region_size, 0)
    x_max = min(cx + region_size, image.shape[1])
    y_min = max(cy - region_size, 0)
    y_max = min(cy + region_size, image.shape[0])
    region_of_interest = hsv_image[y_min:y_max, x_min:x_max]

    # Step 4: Extract the Value channel in the region of interest and threshold for shadows
    value_channel = region_of_interest[:, :, 2]
    _, shadow_mask = cv2.threshold(value_channel, 80, 255, cv2.THRESH_BINARY_INV)

    # Step 5: Expand the shadow mask to match the full image size
    full_shadow_mask = np.zeros_like(ball_mask)
    full_shadow_mask[y_min:y_max, x_min:x_max] = shadow_mask

    # Display the intermediate result - Shadow Mask
    plt.figure(figsize=(5, 5))
    plt.title("Figure 3: Shadow Mask in Region of Interest")
    plt.imshow(full_shadow_mask, cmap='gray')
    plt.axis('off')
    plt.show()

    # Step 6: Find contours in the shadow mask and mark the shadow on the original image
    shadow_contours, _ = cv2.findContours(full_shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_image = image.copy()
    
    for contour in shadow_contours:
        # Draw shadow contour with a distinct color (e.g., red) on the original image
        cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)  # Red color

    # Display the final result with shadow marked
    plt.figure(figsize=(5, 5))
    plt.title("Figure 4: Original Image with Shadow Marked")
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

else:
    print("No bright orange object detected in the image.")




"""Problem 2: Simple Color Image Retrieval"""
"""------------------------------\
    ----------------------------"""

"""Question 1"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def CalNormalizedHSVHist(im, hBinNum, sBinNum, vBinNum):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram for each channel with specified bins
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [hBinNum, sBinNum, vBinNum],
                        [0, 180, 0, 256, 0, 256])
    
    # Normalize the histogram
    hist = hist / np.sum(hist)
    
    # Flatten to a 1-D vector
    return hist.flatten()

# Load images
image_files = ['C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/Elephant1.jpg', \
               'C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/Elephant2.jpg', \
                   'C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/Horse1.jpg', \
                   'C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/Horse2.jpg']
hBinNum, sBinNum, vBinNum = 4, 4, 4

# Compute histograms for each image
histograms = []
for file in image_files:
    image = cv2.imread(file)
    hist = CalNormalizedHSVHist(image, hBinNum, sBinNum, vBinNum)
    histograms.append(hist)

# Plot the histograms
plt.figure(figsize=(10, 8))
for i, hist in enumerate(histograms):
    plt.subplot(2, 2, i+1)
    plt.title(image_files[i].split('/')[-1])
    plt.plot(hist)
    plt.xlabel("Bin")
    plt.ylabel("Percentage of Pixels")
plt.suptitle("Figure 5: Normalized HSV Histograms for Image Database")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""Question 2"""

def histogram_intersection(h, g):
    # Compute histogram intersection
    return np.sum(np.minimum(h, g))

# Retrieval for each image as the query
for query_idx, query_hist in enumerate(histograms):
    similarities = []
    
    # Compute similarity score between the query and each image in the database
    for idx, hist in enumerate(histograms):
        similarity = histogram_intersection(query_hist, hist)
        similarities.append((similarity, idx))
    
    # Sort results by similarity (highest score first)
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Display results
    plt.figure(figsize=(10, 5))
    for rank, (score, img_idx) in enumerate(similarities):
        image = cv2.imread(image_files[img_idx])
        plt.subplot(1, 4, rank + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Rank {rank + 1}\nScore: {score:.2f}")
        plt.axis('off')
    
    plt.suptitle(f"Figure {6 + query_idx}: Retrieval Results for Query {image_files[query_idx]}")
    plt.show()



"""Problem 3: A Simple Watermarking Technique in Wavelet Domain"""
"""------------------------------\
    ----------------------------"""
"""Question 1"""

# Embedding function
def embed_watermark(image, beta=30):
    np.random.seed(42)  # Ensure reproducibility
    # Perform 3-level wavelet decomposition
    coeffs = pywt.wavedec2(image, 'db9', level=3)
    LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

    # Generate a random binary sequence b with length equal to LL3
    b = np.random.randint(0, 2, LL3.shape, dtype=np.uint8)
    # Embedding the binary sequence b into the LL3 subband
    watermarked_LL3 = LL3.copy()
    for i in range(LL3.shape[0]):
        for j in range(LL3.shape[1]):
            if b[i, j] == 1:
                if LL3[i, j] % beta >= 0.25 * beta:
                    watermarked_LL3[i, j] = LL3[i, j] - (LL3[i, j] % beta) + 0.75 * beta
                else:
                    watermarked_LL3[i, j] = (LL3[i, j]-0.25*beta) - ((LL3[i, j]-0.25*beta) % beta) + 0.75 * beta
            else:
                if LL3[i, j] % beta <= 0.75 * beta:
                    watermarked_LL3[i, j] = LL3[i, j] - (LL3[i, j] % beta) + 0.25 * beta
                else:
                    watermarked_LL3[i, j] = (LL3[i, j]+ 0.5 * beta) - ((LL3[i, j] - 0.5 * beta) % beta) + 0.25 * beta

    # Recompose the coefficients with the modified LL3
    watermarked_coeffs = (watermarked_LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))
    watermarked_image = pywt.waverec2(watermarked_coeffs, 'db9')
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    # Calculate the difference image for visualization
    difference_image = 10* cv2.absdiff(image, watermarked_image)  # Scale for better visibility

    return watermarked_image, difference_image, b


# Load the original image
original_image = cv2.imread('C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 6/Lena.jpg', 0)

watermarked_image, difference_image, b = embed_watermark(original_image)
# Display images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Watermarked Image")
plt.imshow(watermarked_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Difference Image (Scaled)")
plt.imshow(difference_image, cmap='gray')
plt.axis('off')
plt.show()

"""Question 2"""

# Extraction function
def extract_watermark(watermarked_image, original_watermark_sequence, beta=30):
    # Perform 3-level wavelet decomposition on the watermarked image
    coeffs = pywt.wavedec2(watermarked_image, 'db9', level=3)
    extracted_LL3, (_, _, _), (_, _, _), (_, _, _) = coeffs

    # Extract watermark sequence from the LL3 subband
    extracted_watermark = np.zeros_like(original_watermark_sequence, dtype=np.uint8)
    for i in range(extracted_LL3.shape[0]):
        for j in range(extracted_LL3.shape[1]):
            if extracted_LL3[i, j] % beta > beta / 2:
                extracted_watermark[i, j] = 1
            else:
                extracted_watermark[i, j] = 0

    # Compare extracted watermark with the original watermark
    matching_bits = np.sum(original_watermark_sequence == extracted_watermark)
    total_bits = original_watermark_sequence.size
    match_percentage = (matching_bits / total_bits) * 100

    return match_percentage


match_percentage = extract_watermark(watermarked_image, b)

# Display message based on matching percentage
if match_percentage == 100:
    print("Perfect match: The extracted watermark is identical to the original.")
elif match_percentage >= 90:
    print("High match: The extracted watermark is highly similar to the original.")
elif match_percentage >= 75:
    print("Moderate match: The extracted watermark is moderately similar to the original.")
else:
    print("Low match: Significant differences between the extracted and original watermark.")

print(f"Percentage of matching bits: {match_percentage:.2f}%")

"""Question 3"""

# Embed watermark with beta = 60
watermarked_image_60, difference_image_60, b = embed_watermark(original_image, beta=60)

# Extract watermark with beta = 60
match_percentage_60 = extract_watermark(watermarked_image_60, b, beta=60)

# Display results for beta = 30
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image (β = 60)")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Watermarked Image (β = 60)")
plt.imshow(watermarked_image_60, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Difference Image (β = 60, Scaled)")
plt.imshow(difference_image_60, cmap='gray')
plt.axis('off')
plt.show()


# Display message based on matching percentage
if match_percentage_60 == 100:
    print("Perfect match: The extracted watermark is identical to the original.")
elif match_percentage_60 >= 90:
    print("High match: The extracted watermark is highly similar to the original.")
elif match_percentage_60 >= 75:
    print("Moderate match: The extracted watermark is moderately similar to the original.")
else:
    print("Low match: Significant differences between the extracted and original watermark.")

print(f"Percentage of matching bits: {match_percentage_60:.2f}%")














