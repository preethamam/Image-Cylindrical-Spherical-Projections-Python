import cv2
import numpy as np

from image2cylindrical_v1 import image2cylindrical_v1
from image2cylindrical_v2 import image2cylindrical_v2
from image2cylindricalConcave import image2cylindricalConcave
from image2spherical_v1 import image2spherical_v1
from image2spherical_v2 import image2spherical_v2

# Inputs
file_name = 'checker.jpg'

# Focal lengths
fx = 50
fy = 50

# Distortion coefficients [k1, k2, k3, p1, p2]
dist_coffs = [0, 0, 0, 0, 0]

if __name__ == '__main__':
    
    # Read image
    image = cv2.imread("checker.jpg")

    # Get image size
    height, width, by_pixs = image.shape

    # Camera intrinsics
    K = np.array([[fx, 0, width/2], 
                [0, fy, height/2],
                [0, 0, 1]])
    
    # Image to cylindrical versions 1 and 2
    img_cyl_v1 = image2cylindrical_v1(image, K, dist_coffs)
    img_cyl_v2 = image2cylindrical_v2(image, K, dist_coffs)
    
    # Image to cylindrical versions 1 and 2
    img_sph_v1 = image2spherical_v1(image, K, dist_coffs)
    img_sph_v2 = image2spherical_v2(image, K, dist_coffs)
    
    # Image to cylindrical concave
    img_cyl_ccv = image2cylindricalConcave(image, K, dist_coffs)
    
    # Image to cylindrical concatenate the images horizontally
    img_v1_horizontal = cv2.hconcat([image, img_cyl_v1, img_cyl_ccv, img_sph_v1])

    # Image to cylindrical concatenate the images horizontally
    img_v2_horizontal = cv2.hconcat([image, img_cyl_v2, img_cyl_ccv, img_sph_v2])
    
    # Display result
    cv2.imshow('Version 1 Projection', img_v1_horizontal)
    cv2.imshow('Version 2 Projection', img_v2_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()