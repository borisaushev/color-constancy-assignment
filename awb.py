import numpy as np 
import cv2


def GrayWorld(image: np.ndarray):
    """
    GrayWorld algorithm for automatic white balance.
    Args:
        image: Image (height, width, 3)
    Returns:
        white_point: Estimated white point multipliers normalized
            by the green channel, shape (3,)
    """
    # white_point = np.sum(np.sum(image, axis = 0), axis = 0)
    white_point = np.mean(image, axis=(0, 1))
    return white_point / white_point[1]


def MaxRGB(image: np.ndarray):
    """
    MaxRGB algorithm for automatic white balance.
    Args:
        image: Image (height, width, 3)
    Returns:
        white_point: Estimated white point multipliers normalized
            by the green channel, shape (3,)
    """
    white_point = np.max(image, axis=(0,1))
    return white_point / white_point[1]


def GrayEdge(image: np.ndarray):
    """
    Gray-Edge style AWB: average RGB on high-gradient pixels (luminance), then
    normalize by the green channel

    Args:
        image: Image (height, width, 3)

    Returns:
        white_point: Shape (3,), illuminant estimate normalized by green.
    """
    luminance = np.average(image, axis=2)
    (gradY, gradX) = np.gradient(luminance)
    gradient = gradY**2 + gradX**2
    threshold = np.percentile(gradient, 95)

    white_point = np.mean(image[gradient > threshold], axis=0)
    return white_point / white_point[1]
