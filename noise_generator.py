import cv2 as cv
import numpy as np

#Step 1 and 2:
def non_uniform_illumination(original_img, brightness):    
    # Step 1:
    # Check if brightness parameter is valid
    if brightness == 'bright':
        alpha = 2.5
        beta = 1
    elif brightness == 'dark':
        alpha = 2
        beta = 0.2
    else:
        raise ValueError('Brightness of non_uniform_illumination must be bright or dark!')

    # Apply Gamma Correction with alpha and beta parameters
    gc_img = pow(original_img, beta)*alpha
    
    # Apply Gaussian Blur to create I0
    blurred_img = cv.GaussianBlur(gc_img, (3,3), sigmaX=0, sigmaY=0, borderType=cv.BORDER_DEFAULT)
    
    # Step 2:
    # Normalize the blurred image I0 to range [0, 1]
    blurred_img = blurred_img.astype('float32')
    blurred_img = (blurred_img - np.min(blurred_img)) / (np.max(blurred_img) - np.min(blurred_img))*1.0
    
    # Calculate average pixel value of original_img (I) and blurred_img (I0) to compute the scaling factor
    avg_pixel_value_I = np.mean(original_img)
    avg_pixel_value_I0 = np.mean(blurred_img)
    scaling_factor = avg_pixel_value_I / avg_pixel_value_I0
    
    # Adjust the average pixel value of I0 to the same as I to create adjusted_I0 (IM0)
    adjusted_I0 = blurred_img*scaling_factor
    
    # Calculate the contrast-adjusted image I1 = 2*IM0 - I0
    ca_img = 2*adjusted_I0 - blurred_img
    
    # Scale the image back to the range [0, 255]
    output = (ca_img - np.min(ca_img)) / (np.max(ca_img) - np.min(ca_img))*255.0
    output = output.astype('uint8')
    
    return output


#Step 3:
def add_gaussian_noise(img):
    #Normalize the contrast-adjusted image I1 to range [0, 1]
    norm_img = img.astype('float32')
    norm_img = (norm_img - np.min(norm_img)) / (np.max(norm_img) - np.min(norm_img))*1.0
    
    #Add Gaussian noise
    mean = 0
    std = 0.005
    noise = np.random.normal(mean, std, norm_img.shape).astype('float32') #Create Gaussian noise using numpy
    noisy_img = norm_img + noise
    
    # Scale the image back to the range [0, 255]
    output = (noisy_img - np.min(noisy_img)) / (np.max(noisy_img) - np.min(noisy_img))*255.0
    output = output.astype('uint8')
    
    return output

if __name__ == '__main__':
    
    img_path = './sample.bmp'
    img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)
    bright_img = non_uniform_illumination(original_img=img, brightness='bright')
    dark_img = non_uniform_illumination(original_img=img, brightness='dark')
    noisy_img = add_gaussian_noise(dark_img)

    cv.imshow('Image', img)
    cv.imshow('Bright Image', bright_img)
    cv.imshow('Dark Image', dark_img)
    cv.imshow('Noisy dark image', noisy_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    