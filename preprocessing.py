import cv2 as cv
import numpy as np

def background_removal(image):
    
    # Binarize the image
    _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)

    # Apply Sobel filter to create an edge map
    sobel_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=-1)
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=-1)
    sobel_y = cv.convertScaleAbs(sobel_y)
    edge_map = cv.addWeighted(sobel_x, 0.3, sobel_y, 0.3, 0)
    edge_map = cv.bitwise_not(edge_map)
    
    # Generate the difference image
    diff_image = np.maximum(binary_image - edge_map, 0)
    
    # Area thresholding using 25x25 Gaussian distribution
    kernel_size = 25
    gaussian_kernel = cv.getGaussianKernel(kernel_size, 1)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # 2D Gaussian kernel
    threshold = cv.filter2D(diff_image, -1, gaussian_kernel).mean()
    _, thresholded_img = cv.threshold(diff_image, threshold, 255, cv.THRESH_BINARY_INV)
    
    # Extract contours
    contours, _ = cv.findContours(thresholded_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    mask = np.full_like(image, 255)
    cv.drawContours(image=mask, contours=contours, contourIdx=-1, color=0, thickness=1)
    
    
            
    return mask

if __name__ == '__main__':
    img = cv.imread('sample.bmp', cv.IMREAD_GRAYSCALE)
    mask = background_removal(img)
    cv.imshow('image', img)
    cv.imshow('mask', mask)
    cv.waitKey(0)
    cv.destroyAllWindows()