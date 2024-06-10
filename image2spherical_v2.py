import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def image2spherical_v2(image, K, DC, interpolate=1):
    """
    Image to spherical projection
    Projects normal image to a spherical warp

    Code author: Preetham Manjunatha
    Github link: https://github.com/preethamam
    Date: 05/04/2024

    Usage: imageSpherical = image2spherical_v2(image, K, DC, interpolate)
    Inputs:
        image - input image
        K - Camera intrinsic matrix (depends on the camera)
        DC - Radial and tangential distortion coefficient [k1, k2, k3, p1, p2]
        interpolate - 0 (no) or 1 (yes)
    Outputs:
        imageSpherical - Warpped image to spherical coordinates
    """

    # Get distortion coefficients
    fx, fy = K[0, 0], K[1, 1]
    k1, k2, k3, p1, p2 = DC

    # Get image size
    ydim, xdim, bypixs = image.shape

    # Get the center of image
    xc, yc = xdim // 2, ydim // 2

    # Create X and Y coordinates grid
    X, Y = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))

    # Perform the cylindrical projection
    theta = (X - xc) / fx
    phi = (Y - yc) / fy

    # Spherical coordinates to Cartesian
    xcap = np.sin(theta) * np.cos(phi)
    ycap = np.sin(phi)
    zcap = np.cos(theta) * np.cos(phi)

    xyz_cap = np.stack((xcap, ycap, zcap), axis=-1)
    xyz_cap = xyz_cap.reshape(-1, 3)

    # Normalized coords
    xyz_cap_norm = (K @ xyz_cap.T).T
    xn = xyz_cap_norm[:, 0] / xyz_cap_norm[:, 2]
    yn = xyz_cap_norm[:, 1] / xyz_cap_norm[:, 2]

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
        imageSpherical = np.zeros_like(image, dtype=image.dtype)
        imageSpherical[mask] = image.ravel()[ind]
    else:
        # Interpolate for each color channel
        imageSpherical = np.zeros_like(image, dtype=float)
        for k in range(image.shape[2]):
            imageSpherical[:, :, k] = map_coordinates(image[:, :, k], [yd - 1, xd - 1], order=3, mode='constant', cval=0)
        imageSpherical = np.uint8(imageSpherical)

    return imageSpherical

