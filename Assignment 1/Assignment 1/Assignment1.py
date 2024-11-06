# -*- coding: utf-8 -*-
"""
Personal Details:
    name: Md. Robiul Islam
    A-number: A02437860
    Email: robiul@ece.ruet.ac.bd or a02437860@aggies.usu.edu
    Assignment Number: 01 (Assignment 1 –Warm-up Exercises)

Created on Mon Sep  2 08:51:53 2024

@author: robiul
I have user spyder ide to run the code

"""

"""Assignment 1 question 1"""
"""------------------------------\
    ----------------------------"""
    
#import necessary python packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

#read original images
pepperIm = cv2.imread("C:/Users/robiul/Documents/OpenCV/peppers.bmp", 1)
pepperIm = cv2.cvtColor(pepperIm, cv2.COLOR_BGR2RGB)
lenaIm = cv2.imread("C:/Users/robiul/Documents/OpenCV/Lena.jpg", 0)


fig, axes = plt.subplots(1, 2)

# pepperIm image
axes[0].imshow(pepperIm)
axes[0].set_title('pepperIm')

# lenaIm image
axes[1].imshow(lenaIm, cmap="gray")
axes[1].set_title('lenaIm')

fig.suptitle("Original Images", size=16)
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




"""Assignment 1 question 2"""
"""------------------------------\
    ----------------------------"""

pepperGrayIm = cv2.cvtColor(pepperIm, cv2.COLOR_RGB2GRAY)
pepperGrayImT = pepperGrayIm.transpose()

nrows, ncols = pepperGrayIm.shape

mid = ncols//2
left_half = pepperGrayIm[:, :mid+(ncols%2)]
right_half = pepperGrayIm[:, mid+(ncols%2):]
pepperGrayImV = np.hstack((right_half, left_half))

pepperGrayImF = np.flipud(pepperGrayIm)

fig, axes = plt.subplots(2, 2, figsize=(30, 30))

# 1st image
axes[0, 0].imshow(pepperGrayIm, cmap='gray')
axes[0, 0].set_title('pepperGrayIm')

# 2nd image
axes[0, 1].imshow(pepperGrayImT, cmap="gray")
axes[0, 1].set_title('pepperGrayImT')

# 3rd image
axes[1, 0].imshow(pepperGrayImV, cmap="gray")
axes[1, 0].set_title('pepperGrayImV')

# 4th image
axes[1, 1].imshow(pepperGrayImF, cmap="gray")
axes[1, 1].set_title('pepperGrayImF')

plt.show()




"""Assignment 1 question 3"""
"""------------------------------\
    ----------------------------"""
    
#Saving the max, min, mean and median variables
maximum = np.max(lenaIm)
minimum = np.min(lenaIm)
mean = np.mean(lenaIm)
median = np.median(lenaIm)


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def FindInfo(image):
    """

    Parameters
    ----------
    image : The image of which various statistics 
    have to be calculated.

    Returns
    -------
    -This function returns the maximum, minimum, mean and median
    of the supplied image.
    -if a color channel image is supplied to this function,
    it will print a message with "This function is only for grayscale image"

    """
    if len(image.shape)==3:
        return print("This function is only for grayscale image")
    else:
        maximum = image[0, 0]
        minimum = image[0, 0]
        sum = 0
        array_1d = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                sum+=image[i, j]
                array_1d.append(image[i, j])
                if maximum<image[i, j]:
                    maximum = image[i, j]
                if minimum>image[i, j]:
                    minimum = image[i, j]
        
        mean = sum/(image.shape[0]*image.shape[1])
        sorted_array = quick_sort(array_1d)
        n = len(sorted_array)
        mid = n//2
        if n%2==0:
            median = (sorted_array[mid].astype(float)+sorted_array[mid-1].astype(float))/2
        else:
            median = sorted_array[mid]
        return maximum, minimum, mean, median

mx, mn, m, med = FindInfo(lenaIm)

#Compare computed statistics with built-in functions stattics values
if mx==maximum:
    if mn==minimum:
        if m ==mean:
            if med == median:
                print("All values matched with the built-in functions values")
            else:
                print("median value haven't match")
        else:
            print("Mean value haven't match")
    else:
        print("minimum value haven't match")
else:
    print("maximum value haven't match")




"""Assignment 1 question 4"""
"""------------------------------\
    ----------------------------"""
    
    
def normalization(image):
    min_v = image.min()
    max_v = image.max()
    
    normalized_image = (image-min_v)/(max_v-min_v)
    
    return normalized_image

normalizedLenaIm = normalization(lenaIm)

#show the normalizedLenaIm image

plt.imshow(normalizedLenaIm, cmap="gray")
plt.title("Normalized Grayscale Image", size=16)
plt.show()

r_perquarter = 512//4
normalizedLenaIm[r_perquarter*1:r_perquarter*2, :] = normalizedLenaIm[r_perquarter*1:r_perquarter*2, :]**1.25#2nd quarter of rows
normalizedLenaIm[r_perquarter*2:r_perquarter*3, :] = normalizedLenaIm[r_perquarter*2:r_perquarter*3, :]**0.25#3rd quarter of rows

processedNormalizedLenaIm = normalizedLenaIm

#saving the processedNormalizedLenaIm image 
cv2.imwrite("C:/ROBIUL/Courses Fall 2024/Computer Vision (CS5680)/Assignment 1/Robiul_processedNormalizedLenaIm.jpg", (processedNormalizedLenaIm*255).astype('uint8'))



"""Assignment 1 question 5"""
"""------------------------------\
    ----------------------------"""

pepperGrayImN = normalization(pepperGrayIm)

mean = pepperGrayImN.mean()
std = pepperGrayImN.std()
threshold = abs(mean-std)

print("Threshold value: ", threshold)

#1st efficient approach

bw1 = np.where(pepperGrayImN>threshold, 1, 0) 


#2nd efficient approach
bw2 = pepperGrayImN-threshold
bw2[bw2<=0]=0 
bw2[bw2>0]=1


#Using built-in function
t, bw3 = cv2.threshold(pepperGrayImN, threshold, 1, cv2.THRESH_BINARY)


if np.all((bw1 == bw2) & (bw2 == bw3)):
    print("Both of my methods worked")
elif bw1==bw3:
    print("My method 1 worked but not my method 2")
elif bw2==bw3:
    print("My method 2 worked but not my method 1")
else:
    print("Both of my methods did not worked")
        
#displaying all figures side-by-side
fig, axes = plt.subplots(1, 3)

# 1st Method's image
axes[0].imshow(bw1, cmap="gray")
axes[0].set_title('my first method')

# 2nd method's image
axes[1].imshow(bw2, cmap="gray")
axes[1].set_title('my second method')

# Built-in method's image
axes[2].imshow(bw3, cmap="gray")
axes[2].set_title('Built-in method')

fig.suptitle("Binary Thresholded Images", size=16)
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

"""Assignment 1 question 6"""
"""------------------------------\
    ----------------------------"""

def GenerateBlurImage(image, n=8):
    """

    Parameters
    ----------
    image : numpy image matrix to be processed here
         image can be colored or graysacle image
    n : int, which represent the block size
        The default is 8.
    Returns
    -------
    blurred_image : image matrix after processing, the blurred image
        
    """
    if len(image.shape)==2:
        n_channel = 1 
    else:
        n_channel =3
    
    n_rows, n_cols = image.shape[0:2]
    blurred_image = np.zeros_like(image)
    
    if n_channel==1:
        for i in range(0, n_rows, n):
            for j in range(0, n_cols, n):
                block = image[i:i+n, j:j+n]
                average_value = np.mean(block)
                blurred_image[i:i+n, j:j+n] = average_value
                
    if n_channel==3:
        for channel in range(3):
            for i in range(0, n_rows, n):
                for j in range(0, n_cols, n):
                    block = image[i:i+n, j:j+n, channel]
                    average_value = np.mean(block)
                    blurred_image[i:i+n, j:j+n, channel] = average_value
    
    return blurred_image
    
pepperImBlur = GenerateBlurImage(pepperIm, 8)
lenaImBlur = GenerateBlurImage(lenaIm, 16)


#displaying all figures side-by-side
fig, axes = plt.subplots(2, 2)

# 1st Method's image
axes[0, 0].imshow(pepperIm)
axes[0, 0].set_title('original pepperIm')
axes[0, 0].set_axis_off()

# 1st Method's image
axes[0, 1].imshow(lenaIm, cmap="gray")
axes[0, 1].set_title('Original lenaIm')
axes[0, 1].set_axis_off()

# 2nd method's image
axes[1, 0].imshow(pepperImBlur)
axes[1, 0].set_title('Blurred pepperIm')
axes[1, 0].set_axis_off()

# Built-in method's image
axes[1, 1].imshow(lenaImBlur, cmap="gray")
axes[1, 1].set_title('Blurred lenaIm')
axes[1, 1].set_axis_off()

fig.suptitle("Original and corresponding blurred Images", size=16)
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()