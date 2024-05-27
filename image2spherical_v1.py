import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def image2spherical_v1(image, K, DC, interpolate=1):
    """
    Image to spherical projection
    Projects normal image to a spherical warp

    Code author: Preetham Manjunatha
    Github link: https://github.com/preethamam
    Date: 05/04/2024

    Usage: imageSpherical = image2spherical_v1(image, K, DC, interpolate)
    Inputs:
        image - input image
        K - Camera intrinsic matrix (depends on the camera)
        DC - Radial and tangential distortion coefficient [k1, k2, k3, p1, p2]
        interpolate - 0 (no) or 1 (yes)
    Outputs:
        imageSpherical - Warpped image to spherical coordinates
    """

    # Input arguments check
    if K.shape[0] < 2:
        raise ValueError('Require camera intrinsic matrix (K).')

    if len(DC) < 5:
        DC = [0, 0, 0, 0, 0]
        interpolate = 1

    if len(DC) < 4:
        interpolate = 1

    # Get distortion coefficients
    fx, fy = K[0, 0], K[1, 1]
    k1, k2, k3, p1, p2 = DC

    # Get image size
    ydim, xdim, bypixs = image.shape

    # Get the center of image
    xc, yc = xdim // 2, ydim // 2

    # Create X and Y coordinates grid
    X, Y = np.meshgrid(np.arange(1, xdim+1), np.arange(1, ydim+1))

    # Perform the cylindrical projection
    theta = (X - xc) / fx
    phi = (Y - yc) / fy

    # Spherical coordinates to Cartesian
    xcap = np.sin(theta) * np.cos(phi)
    ycap = np.sin(phi)
    zcap = np.cos(theta) * np.cos(phi)

    xn = xcap / zcap
    yn = ycap / zcap

    # Radial and tangential distortion
    r = xn**2 + yn**2
    xd_r = xn * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    yd_r = yn * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)

    xd_t = 2 * p1 * xn * yn + p2 * (r**2 + 2 * xn**2)
    yd_t = p1 * (r**2 + 2 * yn**2) + 2 * p2 * xn * yn

    xd = xd_r + xd_t
    yd = yd_r + yd_t

    # Convert to ceil
    xd = np.ceil(fx * xd + xc).astype(int)
    yd = np.ceil(fy * yd + yc).astype(int)

    # Get projections
    if interpolate == 0:
        # Clip coordinates
        mask = (xd > 0) & (xd <= xdim) & (yd > 0) & (yd <= ydim)
        ind = np.ravel_multi_index((yd[mask]-1, xd[mask]-1, np.zeros_like(xd[mask])), image.shape)
        IC1 = np.zeros((ydim, xdim), dtype=np.uint8)
        IC1[mask] = image.ravel()[ind]

        if bypixs == 1:
            imageSpherical = IC1
        else:
            IC2 = np.zeros((ydim, xdim), dtype=np.uint8)
            IC3 = np.zeros((ydim, xdim), dtype=np.uint8)
            IC2[mask] = image.ravel()[ind + 1 * ydim * xdim]
            IC3[mask] = image.ravel()[ind + 2 * ydim * xdim]
            imageSpherical = np.stack([IC1, IC2, IC3], axis=-1)
    else:
        # Interpolate for each color channel
        imageSpherical = np.zeros_like(image, dtype=np.float32)
        for k in range(image.shape[2]):
            imageSpherical[:, :, k] = map_coordinates(image[:, :, k], [yd, xd], order=3, mode='constant', cval=0)

        imageSpherical = np.uint8(imageSpherical)

    return imageSpherical

