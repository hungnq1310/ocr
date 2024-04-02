from pathlib import Path
import pdf2image
import numpy as np
import cv2
import argparse


def bbox2ibox(points):
    min_x, min_y = min(points[:, 0]), min(points[:, 1])
    max_x, max_y = max(points[:, 0]), max(points[:, 1])
    return (int(min_x), int(min_y)), (int(max_x), int(max_y))


def pdf2imgs(path):
    path = Path(path).expanduser().resolve()
    imgs = pdf2image.convert_from_path(str(path))
    imgs = [np.array(x) for x in imgs] if isinstance(imgs, list) else [imgs]
    return imgs


def cv2crop(img, a, b):
    crop = img[a[1] : b[1], a[0] : b[0]]
    return crop


def cv2drawbox(img, a, b):
    img = cv2.rectangle(img, a, b, color=(255, 0, 0), thickness=2)
    return img


def lineDetect(img):
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blank = img * 0
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow("edge", edges)
    # cv2.waitKey()
    # This returns an array of r and theta values
    _ = cv2.HoughLinesP(
        edges,  # Input edge image
        5,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10,  # Max allowed gap between line for joining them
    )
    return blank


# Document Image Rectification
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
