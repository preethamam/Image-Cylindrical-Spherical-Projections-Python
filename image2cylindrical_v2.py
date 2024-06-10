import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def image2cylindrical_v2(image, K, DC, interpolate=1):
    """
    Image to cylindrical projection
    Projects normal image to a cylindrical warp

    Code author: Preetham Manjunatha
    Github link: https://github.com/preethamam
    Date: 05/27/2024

    Usage:
    imageCylindrical = image2cylindrical_v2(image, K, DC, interpolate)

    Inputs:
    image - input image
    K - Camera intrinsic matrix (depends on the camera)
    DC - Radial and tangential distortion coefficient [k1, k2, k3, p1, p2]
    interpolate - 0 (no) or 1 (yes)

    Outputs:
    imageCylindrical - Warpped image to cylindrical coordinates
    """

    # Input arguments check
    if K.size < 2:
        raise ValueError('Require camera intrinsic matrix (K).')

    if len(DC) < 5:
        DC = np.array([0, 0, 0, 0, 0])
        interpolate = 1

    if len(DC) < 4:
        interpolate = 1

    # Get distortion coefficients
    fx, fy = K[0, 0], K[1, 1]
    k1, k2, k3, p1, p2 = DC

    # Get image size
    ydim, xdim, bypixs = image.shape

    # Get the center of image
    xc, yc = xdim / 2, ydim / 2

    # Create X and Y coordinates grid
    X, Y = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))

    # Perform the cylindrical projection
    theta = (X - xc) / fx
    h = (Y - yc) / fy

    # Cylindrical coordinates to Cartesian
    xcap, ycap, zcap = np.sin(theta), h, np.cos(theta)
    xyz_cap = np.stack((xcap, ycap, zcap), axis=-1).reshape(-1, 3)

    # Normalized coords
    xyz_cap_norm = (K @ xyz_cap.T).T
    xn, yn = xyz_cap_norm[:, 0] / xyz_cap_norm[:, 2], xyz_cap_norm[:, 1] / xyz_cap_norm[:, 2]

    # Radial and tangential distortion
    r = xn ** 2 + yn ** 2
    xd_r = xn * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    yd_r = yn * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)

    xd_t = 2 * p1 * xn * yn + p2 * (r ** 2 + 2 * xn ** 2)
    yd_t = p1 * (r ** 2 + 2 * yn ** 2) + 2 * p2 * xn * yn

    xd = xd_r + xd_t
    yd = yd_r + yd_t

    # Reshape to image dimension 2D
    xd = np.ceil(xd).astype(int).reshape(ydim, xdim)
    yd = np.ceil(yd).astype(int).reshape(ydim, xdim)
    
    # Get projections
    if interpolate == 0:
        # Clip coordinates
        mask = (xd > 0) & (xd <= xdim) & (yd > 0) & (yd <= ydim)
        ind = np.ravel_multi_index((yd[mask] - 1, xd[mask] - 1), image.shape[:2])
        IC1 = np.zeros((ydim, xdim), dtype=np.uint8)
        IC1[mask] = image.ravel()[ind]

        if bypixs == 1:
            imageCylindrical = IC1
        else:
            IC2 = np.zeros((ydim, xdim), dtype=np.uint8)
            IC3 = np.zeros((ydim, xdim), dtype=np.uint8)
            IC2[mask] = image.ravel()[ind + ydim * xdim]
            IC3[mask] = image.ravel()[ind + 2 * ydim * xdim]
            imageCylindrical = np.stack((IC1, IC2, IC3), axis=-1)
    else:
        # Initialize array
        imageCylindrical = np.zeros_like(image, dtype=np.float32)

        # Interpolate for each color channel
        for k in range(image.shape[2]):
            imageCylindrical[:, :, k] = map_coordinates(image[:, :, k], [yd, xd], order=3, mode='constant', cval=0)

        # Display the result
        imageCylindrical = np.uint8(imageCylindrical)

    return imageCylindrical

