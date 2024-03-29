import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Step 1:
def non_uniform_illumination(img, kernel_size, brightness):
    
    # Check if brightness parameter is valid
    if brightness == 'bright':
        alpha = 2.5
        beta = 1
    elif brightness == 'dark':
        alpha = 2
        beta = 0.3
    else:
        raise ValueError('Brightness of non_uniform_illumination must be bright or dark!')

    # Apply Gamma Correction with alpha and beta parameters
    gc_img = np.zeros(img.shape, img.dtype)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for channel in range(img.shape[2]):
                gc_img[row, col, channel] = np.clip(pow(img[row, col, channel], beta)*alpha, 0, 255)
    
    # Apply Gaussian Blur
    blurred_img = cv.GaussianBlur(gc_img, kernel_size, sigmaX=0, sigmaY=0, borderType=cv.BORDER_DEFAULT)

    return blurred_img

#Step 2:
def contrast_adjustment(img):

    #Calculate average value of pixels in image
    mean_pixel_value = np.mean(img)

    #Contrast adjustment
    ca_img = np.zeros(img.shape, img.dtype)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for channel in range(img.shape[2]):
                ca_img[row, col, channel] = np.clip((2*mean_pixel_value - img[row, col, channel]), 0, 255)

    return ca_img

#Step 3:
def add_gaussian_noise(img):
    
    #Normalize the contrast-adjusted image
    img = img.astype('float32')
    norm_img = img/255.0
    
    #Add Gaussian noise
    mean = 0
    std = 0.005
    noise = np.random.normal(mean, std, norm_img.shape).astype('float32') #Create Gaussian noise using numpy
    noisy_img = cv.add(norm_img, noise)
    # noisy_img = np.clip((noisy_img*255.0), 0, 255).astype('int')
    
    return noisy_img

if __name__ == '__main__':
    
    img_path = './sample.bmp'
    img = cv.imread(filename=img_path, flags=cv.IMREAD_UNCHANGED)
    blurred_img = non_uniform_illumination(img=img, kernel_size=(3,3), brightness='bright')
    ca_img = contrast_adjustment(blurred_img)
    noisy_img = add_gaussian_noise(ca_img)
    
    plt.figure()
    
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title("Original Image I")
    
    plt.subplot(1,4,2)
    plt.imshow(blurred_img)
    plt.title("I0")
    
    plt.subplot(1,4,3)
    plt.imshow(ca_img)
    plt.title("I1")
    
    plt.subplot(1,4,4)
    plt.imshow(noisy_img)
    plt.title("I2")
    
    plt.show()