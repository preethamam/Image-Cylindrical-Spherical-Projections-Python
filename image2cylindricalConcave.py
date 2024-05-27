import numpy as np
from scipy.ndimage.interpolation import map_coordinates

def image2cylindricalConcave(image, K, DC, interpolate=1):
    """
    Image to cylindrical projection (concave)
    Projects normal image to a cylindrical warp

    Code author: Preetham Manjunatha
    Github link: https://github.com/preethamam
    Date: 05/27/2024

    Usage:
    imageCylindrical = image2cylindricalConcave(image, K, DC, interpolate)

    Inputs:
    image - input image
    K - Camera intrinsic matrix (depends on the camera)
    DC - Radial and tangential distortion coefficient [k1, k2, k3, p1, p2]
    interpolate - 0 (no) or 1 (yes)

    Outputs:
    imageCylindrical - Warpped image to cylindrical coordinates

    Acknowledgements:
    Hammer (Stack Overflow)
    https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection
    """

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
    # Center the point at (0, 0)
    pcX, pcY = X - xc, Y - yc

    # These are your free parameters
    r, omega = xdim, xdim / 2
    z0 = fx - np.sqrt(r ** 2 - omega ** 2)

    zc = (2 * z0 + np.sqrt(4 * z0 ** 2 - 4 * (pcX ** 2 / (fx ** 2) + 1) * (z0 ** 2 - r ** 2))) / (2 * (pcX ** 2 / (fy ** 2) + 1))
    xn = (pcX * zc / fx) + xc
    yn = (pcY * zc / fy) + yc

    # Radial and tangential distortion
    r = xn ** 2 + yn ** 2
    xd_r = xn * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    yd_r = yn * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)

    xd_t = 2 * p1 * xn * yn + p2 * (r ** 2 + 2 * xn ** 2)
    yd_t = p1 * (r ** 2 + 2 * yn ** 2) + 2 * p2 * xn * yn

    xd = xd_r + xd_t
    yd = yd_r + yd_t

    # Convert to ceil
    xd, yd = np.ceil(xd).astype(int), np.ceil(yd).astype(int)

    # Get projections
    if interpolate == 0:
        # Clip coordinates
        mask = (xd > 0) & (xd <= xdim) & (yd > 0) & (yd <= ydim)

        ind = np.ravel_multi_index((yd[mask] - 1, xd[mask] - 1), image.shape[:2])
        imageCylindrical = np.zeros((ydim, xdim, bypixs), dtype=image.dtype)
        imageCylindrical[mask] = image.flat[ind]
    else:
        # Interpolate for each color channel
        imageCylindrical = np.zeros_like(image, dtype=float)
        for k in range(image.shape[2]):
            imageCylindrical[:, :, k] = map_coordinates(image[:, :, k], [yd, xd], order=3, mode='constant', cval=0)

        imageCylindrical = np.uint8(imageCylindrical)

    return imageCylindrical

