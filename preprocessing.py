import cv2 as cv
import numpy as np

def background_removal(image):
    
    # Binarize the image
    _, binary_image = cv.threshold(image, 200, 255, cv.THRESH_BINARY_INV)

    # Apply Sobel filter to create an edge map
    sobel_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=-1)
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=-1)
    sobel_y = cv.convertScaleAbs(sobel_y)
    edge_map = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    edge_map = cv.bitwise_not(edge_map)
    
    # Generate the difference image
    diff_image = np.maximum(binary_image - edge_map, 0)
    
    # Area thresholding using 25x25 Gaussian distribution
    thresholded_img = cv.adaptiveThreshold(diff_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 1)
    
    # Extract contours
    contours, _ = cv.findContours(image=thresholded_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresholded_img)
    cv.drawContours(image=mask, contours=contours, contourIdx=-1, color=255, thickness=cv.FILLED)
    
    # Apply morphological opening to separate areas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    # Apply component labeling to remove background
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask)

    # Find the largest component (excluding background)
    largest_area = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1
    
    # Create a mask with only the largest area
    mask = np.where(labels == largest_area, 255, 0).astype(np.uint8)

    # Morphological erosion
    mask = cv.erode(mask, kernel, iterations=1)
    
    # Hole filling
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    return mask

def rotate_image(mask_image):

    # Calculate the center coordinate (gx, gy)
    coords = np.argwhere(mask_image)
    gx, gy = np.mean(coords, axis=0)
    print(gx, gy)

    # Initialize summations
    sum_xy = sum_x2 = sum_y2 = sum_i = 0

    for x, y in coords:
        i = mask_image[x, y]
        sum_xy += (y - gy) * (x - gx) * i
        sum_x2 += (x - gx) ** 2 * i
        sum_y2 += (y - gy) ** 2 * i
        sum_i += i
    
    C11 = sum_y2 / sum_i
    C12 = sum_xy / sum_i
    C22 = sum_x2 / sum_i

    # Calculate theta
    if C11 > C22:
        theta = np.arctan((C11 - C22 + np.sqrt((C11 - C22)**2 + 4*(C12 ** 2))) 
                          / (-2 * C12))
    else: # C11 <= C22
        theta = np.arctan((-2 * C12)
                          / (C22 - C11 + np.sqrt((C22 - C11)**2 + 4*(C12 ** 2))))
    
    # Rotate the image
    h, w = mask_image.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, np.degrees(theta)+90, 1.0)
    rotated_image = cv.warpAffine(mask_image, rotation_matrix, (w, h))
    
    return rotated_image

if __name__ == '__main__':
    img = cv.imread('sample.bmp', cv.IMREAD_GRAYSCALE)
    mask = background_removal(img)
    rotated_image = rotate_image(mask)
    cv.imshow('Original Image', img)
    cv.imshow('Mask Image', mask)
    cv.imshow('Rotated Image', rotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()