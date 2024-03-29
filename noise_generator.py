import cv2 as cv
import numpy as np

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
    gc_img = np.clip(pow(img, beta)*alpha, 0, 255).astype('uint8')
    
    # Apply Gaussian Blur
    blurred_img = cv.GaussianBlur(gc_img, kernel_size, sigmaX=0, sigmaY=0, borderType=cv.BORDER_DEFAULT)

    return blurred_img

#Step 2:
def contrast_adjustment(img):
    #Calculate average value of pixels in image
    mean_pixel_value = np.mean(img)

    #Contrast adjustment
    ca_img = np.clip((2*mean_pixel_value - img), 0, 255).astype('uint8')

    return ca_img

#Step 3:
def add_gaussian_noise(img):
    #Normalize the contrast-adjusted image
    norm_img = img.astype('float32')/255.0
    
    #Add Gaussian noise
    mean = 0
    std = 0.005
    noise = np.random.normal(mean, std, norm_img.shape).astype('float32') #Create Gaussian noise using numpy
    noisy_img = np.clip((norm_img + noise)*255.0, 0, 255).astype('uint8')
    
    return noisy_img

if __name__ == '__main__':
    
    img_path = './sample.bmp'
    img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)
    blurred_img = non_uniform_illumination(img=img, kernel_size=(3,3), brightness='bright')
    ca_img = contrast_adjustment(blurred_img)
    noisy_img = add_gaussian_noise(ca_img)
    
    cv.imshow('Image', noisy_img)
    cv.waitKey(0)
    cv.destroyAllWindows()