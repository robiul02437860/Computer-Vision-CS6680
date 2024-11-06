# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:14:32 2024
Personal Details:
    name: Md. Robiul Islam
    A-number: A02437860
    Email: robiul@ece.ruet.ac.bd or a02437860@aggies.usu.edu
    Assignment Number 04: Filter Techniques in the Frequency Domain

@author: robiul
"""
#import necessary python packages
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt
import pywt  
from skimage.metrics import mean_squared_error
from skimage.util import random_noise
from skimage import img_as_ubyte


"""Problem I: Exercises on Low-pass and High-pass Filters in the Frequency Domain"""
"""Question 1"""
"""------------------------------\
    ----------------------------"""

# Load the image and convert it to grayscale
Sample = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Sample.jpg", 0)

# Get image dimensions
M, N = Sample.shape

sigma_x = 20
sigma_y = 70

#Step1: Create the low-pass Gaussian Filter
H = np.zeros((M, N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        d_x = (u-M/2)**2/(2*sigma_x**2)
        d_y = (v-N/2)**2/(2*sigma_y**2)
        H[u, v] = np.exp(-(d_x+d_y))


# Step 2: Apply Fourier Transform to the image
image_fft = np.fft.fftshift(np.fft.fft2(Sample))

# Step 3: Apply the Gaussian filter (element-wise multiplication in frequency domain)
filtered_fft = image_fft * H

# Step 4: Apply inverse Fourier transform to obtain the filtered image
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
filtered_image = np.abs(filtered_image)


# Plotting the results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(Sample, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gaussian Low-Pass Filter
plt.subplot(1, 3, 2)
plt.imshow(H, cmap='gray')
plt.title('Gaussian Low-Pass Filter')
plt.axis('off')

# Filtered Image
plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()


"""Problem I: Exercises on Low-pass and High-pass Filters in the Frequency Domain"""
"""Question 2"""
"""------------------------------\
    ----------------------------"""

#step-1: Butterworth High Pass Filter
HPF = np.zeros((M, N), dtype=np.float32)
D0 = 50
n = 2
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        HPF[u,v] = 1 / (1 + (D0/D)**(2*n))
        

# Step 2: Apply Fourier Transform to the image
image_fft = np.fft.fftshift(np.fft.fft2(Sample))

# Step 3: Apply the Gaussian filter (element-wise multiplication in frequency domain)
filtered_fft = image_fft * HPF

# Step 4: Apply inverse Fourier transform to obtain the filtered image
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
filtered_image = np.abs(filtered_image)


# Plotting the results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(Sample, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Gaussian Low-Pass Filter
plt.subplot(1, 3, 2)
plt.imshow(HPF, cmap='gray')
plt.title('Butterworth High-Pass Filter')
plt.axis('off')

# Filtered Image
plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()


"""Problem II: Exercise on Certain Operations in the Frequency Domain"""
"""Question 1"""
"""------------------------------\
    ----------------------------"""
# Load the image and convert it to grayscale
Sample = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Sample.jpg", 0)
Capital = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Capitol.jpg", 0)

# Apply Fourier Transform and shift on Sample image
Sample_fft = np.fft.fftshift(np.fft.fft2(Sample))

# Magnitude and phase
Sample_magnitude = 20*np.log(np.abs(Sample_fft)) #log transform and scaling for display purpose
Sample_phase = np.angle(Sample_fft)

# Apply Fourier Transform and shift on Capital Image
Capital_fft = np.fft.fftshift(np.fft.fft2(Capital))

# Magnitude and phase
Capital_magnitude = 20*np.log(np.abs(Capital_fft)) #log transform and scaling for display purpose
Capital_phase = np.angle(Capital_fft)

# Plotting the results
plt.figure(figsize=(12, 8))

# Sample image - Magnitude
plt.subplot(2, 2, 1)
plt.imshow(Sample_magnitude, cmap='gray')
plt.title('Sample Image - Fourier Magnitude (Log Transformed)')
plt.axis('off')

# Sample image - Phase
plt.subplot(2, 2, 2)
plt.imshow(Sample_phase, cmap='gray')
plt.title('Sample Image - Fourier Phase')
plt.axis('off')

# Capital image - Magnitude
plt.subplot(2, 2, 3)
plt.imshow(Capital_magnitude, cmap='gray')
plt.title('Capital Image - Fourier Magnitude (Log Transformed)')
plt.axis('off')

# Capital image - Phase
plt.subplot(2, 2, 4)
plt.imshow(Capital_phase, cmap='gray')
plt.title('Capital Image - Fourier Phase')
plt.axis('off')

plt.tight_layout()
plt.show()


"""Problem II: Exercise on Certain Operations in the Frequency Domain"""
"""Question 2"""
"""------------------------------\
    ----------------------------"""

# Apply Fourier Transform and shift
Sample_fft = np.fft.fft2(Sample)

# Magnitude and phase
Sample_magnitude = np.abs(Sample_fft)
Sample_phase = np.angle(Sample_fft)

# Apply Fourier Transform and shift
Capital_fft = np.fft.fft2(Capital)

# Magnitude and phase
Capital_magnitude = np.abs(Capital_fft)
Capital_phase = np.angle(Capital_fft)

reconstructed_Sample = Capital_magnitude*np.exp(1j*Sample_phase)
reconstructed_Sample = np.fft.ifft2(reconstructed_Sample)
reconstructed_Sample_mag = np.abs(reconstructed_Sample)

reconstructed_Capital = Sample_magnitude*np.exp(1j*Capital_phase)
reconstructed_Capital = np.fft.ifft2(reconstructed_Capital)
reconstructed_Capital_mag = np.abs(reconstructed_Capital)

# Plotting the results
plt.figure(figsize=(12, 8))

# Sample image - Magnitude
plt.subplot(1, 2, 1)
plt.imshow(reconstructed_Capital_mag, cmap='gray')
plt.title('Reconstructed Capital Image')
plt.axis('off')

# Sample image - Phase
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_Sample_mag, cmap='gray')
plt.title('Reconstructed Sample Image')
plt.axis('off')

plt.tight_layout()
plt.show()


"""Problem III: Remove Additive Cosine Noise"""
"""------------------------------\
    ----------------------------"""

boy_noisy = Image.open("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/boy_noisy.gif")
boy_noisy = np.array(boy_noisy)

# Step 1: Compute the centered DFT of the noisy image
def centered_dft(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)  # Shift the zero frequency to the center
    return dft_shifted

centered_dft_image = centered_dft(boy_noisy)


# Step 2: Compute magnitude and find the eight non-center locations corresponding to the four largest distinct magnitudes
def find_largest_magnitudes(magnitude, num_largest=4):
    """
    Finds the eight non-center locations corresponding to the four largest distinct magnitudes
    in the DFT image.
    """
        
    # Get image center to ignore the DC component
    center = np.array(magnitude.shape) // 2
    magnitude[center[0], center[1]] = 0  # Ignore the center (DC component)

    # Flatten and sort the magnitude spectrum
    flat_magnitude = magnitude.ravel()
    sorted_indices = np.argsort(flat_magnitude)[::-1]  # Sort in descending order
    
    # Collect the largest distinct magnitudes
    distinct_coords = []
    count = 0
    visited = set()

    for index in sorted_indices:
        if count >= num_largest:  # Stop when we find the required number of distinct pairs
            break
        
        coord = np.unravel_index(index, magnitude.shape)
        x, y = coord
        
        # Check if the symmetric counterpart is already visited
        sym_x, sym_y = magnitude.shape[0] - x, magnitude.shape[1] - y
        if coord not in visited and (sym_x, sym_y) not in visited:
            distinct_coords.append(coord)
            distinct_coords.append((sym_x, sym_y))  # Add its symmetric counterpart
            visited.add(coord)
            visited.add((sym_x, sym_y))
            count += 1
    
    return distinct_coords


magnitude = np.abs(centered_dft_image)
largest_freq_coords = find_largest_magnitudes(magnitude)


# Step 3: Replace the value at each of the locations in step 2 with the average of neighbors
def replace_with_neighbors(dft_image, coords):
    for x, y in coords:
        # Get the 8 neighbors of (x, y)
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    neighbors.append(dft_image[x + i, y + j])
        
        # Replace the value with the average of the neighbors
        dft_image[x, y] = np.mean(neighbors)
    
    return dft_image

modified_dft_image = replace_with_neighbors(centered_dft_image.copy(), largest_freq_coords)

# Step 4: Take inverse DFT and display original/restored images
def inverse_dft(dft_image):
    return np.abs(np.fft.ifft2(dft_image))  # Inverse FFT and take the magnitude

restored_image = inverse_dft(modified_dft_image)


# Plot original noisy and restored images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original noisy image
axs[0].imshow(boy_noisy, cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original Noisy Image')

# Display the restored image
axs[1].imshow(restored_image, cmap='gray')
axs[1].axis('off')
axs[1].set_title('Restored Image')

plt.tight_layout()
plt.show()


"""Problem III: Remove Additive Cosine Noise"""
"""Question 5 (1, 2, 3, 4)"""
"""------------------------------\
    ----------------------------"""

# Process images for different numbers of largest magnitudes
num_magnitudes = [2, 3, 5, 6]
restored_images = []

for num in num_magnitudes:
    # Find largest magnitudes
    largest_freq_coords = find_largest_magnitudes(magnitude.copy(), num)
    
    # Replace values with neighbors
    modified_dft_image = replace_with_neighbors(centered_dft_image.copy(), largest_freq_coords)
    
    # Inverse DFT to reconstruct the image
    restored_image = inverse_dft(modified_dft_image)
    
    # Store the restored image for plotting
    restored_images.append(restored_image)

# Step 5: Display all restored images side-by-side
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

titles = [
    'Restored Image (2 largest)',
    'Restored Image (3 largest)',
    'Restored Image (5 largest)',
    'Restored Image (6 largest)'
]

for i in range(4):
    axs[i].imshow(restored_images[i], cmap='gray')
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

"""Question 6"""

print("In figure 5, replacing the four largest magnitudes (and their symmetric pairs), the primary cosine noise artifacts was significantly reduced")
print("In figure 6, least amount of noise removed\
 Since only the two largest magnitudes components are targeted. There still exists some artifacts")
print("three largest magnitudes will reduce the noise further compared to the image using only two magnitudes.")
print("The image showed more noise reduction than the previous ones with increased number of largest magnitude replacement, \
      some finer details has started to be lost due to the additional frequency modifications. ")
print("With the increase number of largest magnitude replacement, there is whitist trend in the image.")



"""Problem IV: Preliminary Wavelet Transform"""
"""Question 1"""
"""------------------------------\
    ----------------------------"""

Lena = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Lena.jpg", 0)

max_level = pywt.dwt_max_level(data_len=min(Lena.shape), filter_len=pywt.Wavelet('db2').dec_len)

coeffs = pywt.wavedec2(Lena, wavelet='db2', mode='periodization', level=max_level)
restored_lena = pywt.waverec2(coeffs, wavelet='db2', mode='periodization')
restored_lena = np.clip(restored_lena, 0, 255).astype(np.uint8)

# Step 5: Compare the original and restored images
# Use mean squared error (MSE) to check if the images are the same
mse_value = mean_squared_error(Lena, restored_lena)

if mse_value == 0:
    print("The original image and the restored image are exactly the same!")
else:
    print(f"The original image and the restored image are NOT the same. MSE: {mse_value}")

"""Question 2"""

# A function to apply inverse wavelet transform and display the image
def reconstruct_and_display(coeffs, title):
    # Reconstruct the image using the modified coefficients
    restored_image = pywt.waverec2(coeffs, wavelet='db2', mode='periodization')
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    
    # Display the restored image
    plt.imshow(restored_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Perform 3-level wavelet decomposition using 'db2' wavelet
coeffs = pywt.wavedec2(Lena, wavelet='db2', mode='periodization', level=3)
cA3, l3, l2, l1 = coeffs

# a) Set 16 values of each 4x4 block in the approximation subband to the average
approx_coeffs = coeffs[0].copy()  # Extract approximation coefficients (LL subband)

for i in range(0, approx_coeffs.shape[0], 4):
    for j in range(0, approx_coeffs.shape[1], 4):
        block = approx_coeffs[i:i+4, j:j+4]
        avg_value = np.mean(block)
        approx_coeffs[i:i+4, j:j+4] = avg_value

coeffs_a = [approx_coeffs]
# Reconstruct the coefficient list
for co in coeffs[1:]:
    coeffs_a.append(co)
reconstruct_and_display(coeffs_a, "Image with 4x4 Block Averaged Approximation Subband")


# b) Set first level vertical detail coefficients (LH) to 0
coeffs_b = coeffs.copy()  # Copy the original coefficients
coeffs_b[-1] = (coeffs_b[-1][0], np.zeros_like(coeffs_b[-1][1]), coeffs_b[-1][2])  # Set LH to 0
reconstruct_and_display(coeffs_b, "Image with 1st Level Vertical Detail (LH) Set to 0")

# c) Set second level horizontal detail coefficients (HL) to 0
coeffs_c = coeffs.copy()  # Copy the original coefficients
coeffs_c[-2] = (np.zeros_like(coeffs_c[-2][0]), coeffs_c[-2][1], coeffs_c[-2][2])  # Set HL to 0
reconstruct_and_display(coeffs_c, "Image with 2nd Level Horizontal Detail (HL) Set to 0")

# d) Set third level diagonal detail coefficients (HH) to 0
coeffs_d = coeffs.copy()  # Copy the original coefficients
coeffs_d[-3] = (coeffs_d[-3][0], coeffs_d[-3][1], np.zeros_like(coeffs_d[-3][2]))  # Set HH to 0
reconstruct_and_display(coeffs_d, "Image with 3rd Level Diagonal Detail (HH) Set to 0")

"""Question 3"""

print("Image from Operation (a):")
print("This image appears blocky and has lost fine details")
print("Reason: By averaging the values of 4×4 non-overlapping blocks in the approximation \
      subband (which captures the low-frequency information), we are reducing the detail and smoothing the entire image")

print("Image from Operation (b):")
print("This image has blurred or softened vertical features, particularly at the first decomposition level")
print("Reason: The first level vertical detail coefficients (LH subband) capture vertical high-frequency information. \
      Setting these coefficients to zero removes fine vertical details.")
      
print("Image from Operation (c):")
print("This image has blurred or softened horizontal features at the second decomposition level.")
print("Reason: By setting the second level horizontal detail coefficients (HL subband) to zero, \
      we eliminate horizontal details at that frequency band.")
      
print("Image from Operation (a):")
print("The image loses sharp diagonal features")
print("Reason: The third level diagonal detail coefficients (HH subband) capture diagonal high-frequency details at the coarsest scale. \
      Setting these coefficients to zero removes diagonal edge information in lower-frequency ranges, but since the operation only affects \
          the coarsest level, the impact is less noticeable compared to the vertical and horizontal modifications.")
          
          

"""Problem V: A Simple Solution to Remove Gaussian White Noise"""

"""------------------------------\
    ----------------------------"""

# Load the Lena image
lena = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Lena.jpg", 0)

# Function to add Gaussian White noise
def add_gaussian_white_noise(image, mean=0, variance=0.01):
    noisy_lena_im = random_noise(image, mode='gaussian', mean=0, var=0.01)*255
    return noisy_lena_im.astype(np.uint8)


# Add Gaussian white noise with 0 mean and 0.01 variance
noisy_lena = add_gaussian_white_noise(lena)

# Save the noisy image as 'NoisyLena.bmp'
cv2.imwrite("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/NoisyLena.bmp", noisy_lena)


"""Denoising Method 1"""

# Step 1: Load the noisy Lena image
noisy_lena = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/NoisyLena.bmp", 0)
noisy_lena = noisy_lena.astype(np.float32)

# # Step 2: Apply a 3-level “db2” wavelet decomposition
# coeffs = pywt.wavedec2(noisy_lena, wavelet='db2', mode='periodization', level=3)


# Step 3: Estimate noise standard deviation for the 1st-level diagonal wavelet subband (HH1)
# Helper function for noise estimation (Step 3)
def estimate_noise(HH):
    """Estimate the noise standard deviation using the median absolute deviation (MAD)."""
    return np.median(np.abs(HH)) / 0.6745

# HH1 = coeffs[-1][2]  # HH1 corresponds to coeffs[-1][2]
# sigma_1 = estimate_noise(HH1)

# Step 4: Compute the adaptive threshold for the 1st-level wavelet subbands
def adaptive_thresold(M, sigma):
    return sigma * np.sqrt(2 * np.log(M))

# M1 = (coeffs[-1][0].size)+(coeffs[-1][1].size)+(coeffs[-1][2].size)  # Number of coefficients in each 1st-level subband
# t1 = adaptive_thresold(M1, sigma_1)

# Step 5: Apply soft thresholding to 1st-level subbands (LH1, HL1, HH1)
def soft_threshold(fij, t):
    """
    Apply custom soft thresholding as per the formula provided.
    If fij >= t, then fij' = fij - t
    If fij <= -t, then fij' = fij + t
    If |fij| < t, then fij' = 0
    """
    fij_prime = np.zeros_like(fij)
    fij_prime[fij >= t] = fij[fij >= t] - t
    fij_prime[fij <= -t] = fij[fij <= -t] + t
    fij_prime[np.abs(fij) < t] = 0
    return fij_prime

def denoise_method_1(noisy_lena):
    # Step 2: Apply a 3-level “db2” wavelet decomposition
    coeffs = pywt.wavedec2(noisy_lena, wavelet='db2', mode='periodization', level=3)
    
    
    # Step 3: Estimate noise standard deviation for the 1st-level diagonal wavelet subband (HH1)
    LH1, HL1, HH1 = coeffs[-1]  # HH1 corresponds to coeffs[-1][2]
    sigma_1 = estimate_noise(HH1)
    
    # Step 4: Compute the adaptive threshold for the 1st-level wavelet subbands
    M1 = LH1.size + HL1.size + HH1.size # Number of coefficients in each 1st-level subband
    t1 = adaptive_thresold(M1, sigma_1)
    
    # Step 5: Apply soft thresholding to 1st-level subbands (LH1, HL1, HH1)
    coeffs[-1] = (soft_threshold(LH1, t1), soft_threshold(HL1, t1), soft_threshold(HH1, t1))
    
    # Step 6: Repeat steps 3, 4, and 5 for the 2nd-level subbands (LH2, HL2, HH2)
    LH2, HL2, HH2 = coeffs[-2]
    sigma_2 = estimate_noise(HH2)
    M2 = LH2.size + HL2.size + HH2.size
    t2 = adaptive_thresold(M2, sigma_2)
    coeffs[-2] = (soft_threshold(LH2, t2), soft_threshold(HL2, t2), soft_threshold(HH2, t2))
    
    # Step 7: Repeat steps 3, 4, and 5 for the 3rd-level subbands (LH3, HL3, HH3)
    LH3, HL3, HH3 = coeffs[-3]
    sigma_3 = estimate_noise(HH3)
    M3 = LH3.size + HL3.size + HH3.size
    t3 = adaptive_thresold(M3, sigma_3)
    coeffs[-3] = (soft_threshold(LH3, t3), soft_threshold(HL3, t3), soft_threshold(HH3, t3))
    
    
    # Step 8: Apply inverse wavelet transform to get the denoised image
    denoised_lena = pywt.waverec2(coeffs, wavelet='db2', mode='periodization')
    denoised_lena = np.clip(denoised_lena, 0, 255).astype(np.uint8)
    return denoised_lena


denoised_lena = denoise_method_1(noisy_lena)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(noisy_lena, cmap='gray')
ax1.set_title("Noisy Lena")
ax1.axis('off')

ax2.imshow(denoised_lena, cmap='gray')
ax2.set_title("Denoised Lena")
ax2.axis('off')

plt.show()


"""Denoising Method 2"""

# Step 1: Load the noisy Lena image
noisy_lena = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/NoisyLena.bmp", 0)
noisy_lena = noisy_lena.astype(np.float32)


# Step 3: Estimate noise standard deviation for the 1st-level wavelet subbands 
# Helper function for noise estimation (Step 3)
def estimate_noise(combined_co):
    """Estimate the noise standard deviation using the median absolute deviation (MAD)."""
    median_all = np.median(np.abs(combined_co))
    return median_all / 0.6745



def denoise_method_2(noisy_lena):
    # Step 2: Apply a 3-level “db2” wavelet decomposition
    coeffs = pywt.wavedec2(noisy_lena, wavelet='db2', mode='periodization', level=3)
    
    # Step 3: Estimate noise standard deviation for the 1st-level wavelet subbands
    LH1, HL1, HH1 = coeffs[-1]  # All co-efficients corresponds to coeffs[-1]
    combined = np.concatenate([LH1.flatten(), HL1.flatten(), HH1.flatten()])
    sigma_1 = estimate_noise(combined)
    
    # Step 4: Compute the adaptive threshold for the 1st-level wavelet subbands
    # Number of coefficients in 1st-level subbands
    M1 = LH1.size + HL1.size + HH1.size 
    t1 = adaptive_thresold(M1, sigma_1)
    
    # Step 5: Apply soft thresholding to 1st-level subbands (LH1, HL1, HH1)
    coeffs[-1] = (soft_threshold(LH1, t1), soft_threshold(HL1, t1), soft_threshold(HH1, t1))
    
    # Step 6: Repeat steps 3, 4, and 5 for the 2nd-level subbands (LH2, HL2, HH2)
    LH2, HL2, HH2 = coeffs[-2]
    combined = np.concatenate([LH2.flatten(), HL2.flatten(), HH2.flatten()])
    sigma_2 = estimate_noise(combined)
    M2 = LH2.size + HL2.size + HH2.size 
    t2 = adaptive_thresold(M2, sigma_2)
    coeffs[-2] = (soft_threshold(LH2, t2), soft_threshold(HL2, t2), soft_threshold(HH2, t2))
    
    # Step 7: Repeat steps 3, 4, and 5 for the 3rd-level subbands (LH3, HL3, HH3)
    LH3, HL3, HH3 = coeffs[-3]
    combined = np.concatenate([LH2.flatten(), HL2.flatten(), HH2.flatten()])
    sigma_3 = estimate_noise(combined)
    M3 = LH3.size + HL3.size + HH3.size 
    t3 = adaptive_thresold(M3, sigma_3)
    coeffs[-3] = (soft_threshold(LH3, t3), soft_threshold(HL3, t3), soft_threshold(HH3, t3))
    
    
    # Step 8: Apply inverse wavelet transform to get the denoised image
    denoised_lena = pywt.waverec2(coeffs, wavelet='db2', mode='periodization')
    denoised_lena = np.clip(denoised_lena, 0, 255).astype(np.uint8)
    return denoised_lena

denoised_lena = denoise_method_2(noisy_lena)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(noisy_lena, cmap='gray')
ax1.set_title("Noisy Lena")
ax1.axis('off')

ax2.imshow(denoised_lena, cmap='gray')
ax2.set_title("Denoised Lena")
ax2.axis('off')

plt.show()



# PSNR Calculation function
def compute_mse(image1, image2):
    """Compute Mean Squared Error between two images."""
    return np.mean((image1 - image2) ** 2)

def compute_psnr(original_image, denoised_image):
    """Compute PSNR between original image and transformed image."""
    mse = compute_mse(original_image, denoised_image)
    if mse == 0:
        return float('inf')  # If MSE is 0, PSNR is infinite
    max_pixel_value = 255.0  
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)      # R =255 in case of image
    return psnr

# Load original and noisy images
original_img = cv2.imread("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/Lena.jpg", 0)
noisy_img = cv2.imread('C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 4/NoisyLena.bmp', 0)

# Denoising Method 1 and Method 2 (Assuming you already have these functions from previous tasks)
denoised_img_1 = denoise_method_1(noisy_img)  # Method 1
denoised_img_2 = denoise_method_2(noisy_img)  # Method 2

# PSNR Calculations
psnr_noisy = compute_psnr(original_img, noisy_img)
psnr_denoised_1 = compute_psnr(original_img, denoised_img_1)
psnr_denoised_2 = compute_psnr(original_img, denoised_img_2)
psnr_between_denoised = compute_psnr(denoised_img_1, denoised_img_2)

# Display noisy image and denoised images side-by-side
plt.figure(figsize=(15, 5))

# Plot noisy image
plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB")

# Plot denoised image using Method 1
plt.subplot(1, 3, 2)
plt.imshow(denoised_img_1, cmap='gray')
plt.title(f"Denoised Image - Method 1\nPSNR: {psnr_denoised_1:.2f} dB")

# Plot denoised image using Method 2
plt.subplot(1, 3, 3)
plt.imshow(denoised_img_2, cmap='gray')
plt.title(f"Denoised Image - Method 2\nPSNR: {psnr_denoised_2:.2f} dB")

# Show the plot
plt.tight_layout()
plt.show()

# Output PSNR values and major visual differences
print(f"PSNR between original Lena and Noisy Lena: {psnr_noisy:.2f} dB")
print(f"PSNR between original Lena and Denoised Lena (Method 1): {psnr_denoised_1:.2f} dB")
print(f"PSNR between original Lena and Denoised Lena (Method 2): {psnr_denoised_2:.2f} dB")
print(f"PSNR between denoised image from method 1 and Method 2: {psnr_between_denoised:.2f} dB")

# Output visual differences
print("Visual differences:")
if psnr_denoised_1 > psnr_denoised_2:
    print("Denoising Method 1 provides better image quality with less noise compared to Method 2.")
else:
    print("Denoising Method 2 provides better image quality with less noise compared to Method 1.")







