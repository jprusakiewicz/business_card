# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from PIL import Image
from io import BytesIO
import io


# construct the argument parser and parse the arguments
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def find_best_ratio(photo_path, ratio_parameter: int, PolyDP_ratio=0.02):
    image = cv2.imread(photo_path)
    ratio = image.shape[0] / ratio_parameter
    orig = image.copy()
    # image = decrease_brightness(image, 8)
    image = imutils.resize(image, height=ratio_parameter)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    gray_blurred = cv2.medianBlur(gray, 7)
    #gray_blurred = cv2.medianBlur(gray, kernel, 7)
    #gray_blurred = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
     #gray_blurred = cv2.adaptiveThreshold(gray_blurred,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,4)
    #edged = cv2.Canny(gray_blurred, canny_thresh_l, canny_thresh_h)
    edged = auto_canny(gray_blurred)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    #closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
    # loop over the contours
    screenCnt = False
    contour_area = 0
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, PolyDP_ratio * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            contour_area = cv2.contourArea(c)
            #print(contour_area/(image.shape[0]*image.shape[1]))
            quality_ratio = contour_area/(image.shape[0]*image.shape[1])
            break

    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    except Exception:
        #print('no contours')
        #cv2.imshow("Original", imutils.resize(orig, height=650))
        #cv2.waitKey(0)
        return None
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    t = threshold_local(warped, 11, offset=8, method="mean")
    #t = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    warped = (warped > t).astype("uint8") * 255

    #cv2.imshow("Original", imutils.resize(gray_blurred, height=650))
    #cv2.imshow("Scanned", imutils.resize(warped, height=650))
    #cv2.imshow("Edged", imutils.resize(edged, height=650))
    #cv2.waitKey(0)
    img = Image.fromarray(warped, "L")
    img.save("image_to_send_to_ocr.jpg", "JPEG")
    with open('image_to_send_to_ocr.jpg', 'rb') as f:
        bc = f.read()
    dat = {'file': bc}
    #return dat
    return contour_area

def show_segment(photo_path, ratio_parameter: int, PolyDP_ratio=0.02):
    global quality_ratio
    image = cv2.imread(photo_path)
    ratio = image.shape[0] / ratio_parameter
    orig = image.copy()
    # image = decrease_brightness(image, 8)
    image = imutils.resize(image, height=ratio_parameter)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    gray_blurred = cv2.medianBlur(gray, 7)
    #gray_blurred = cv2.medianBlur(gray, kernel, 7)
    #gray_blurred = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
     #gray_blurred = cv2.adaptiveThreshold(gray_blurred,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,4)
    #edged = cv2.Canny(gray_blurred, canny_thresh_l, canny_thresh_h)
    edged = auto_canny(gray_blurred)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    #closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
    # loop over the contours
    screenCnt = False
    contour_area = 0
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, PolyDP_ratio * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            contour_area = cv2.contourArea(c)
            quality_ratio = contour_area/(image.shape[0]*image.shape[1])
            break

    if screenCnt is False or quality_ratio < 0.15:
        print('trying to close contour')
        edged = cv2.Canny(gray_blurred, 10, 300)
        closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:]
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, PolyDP_ratio * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                contour_area = cv2.contourArea(c)
                quality_ratio = contour_area / (image.shape[0] * image.shape[1])
                print(quality_ratio)
                break
            if screenCnt is False or quality_ratio < 0.15:
                print('DUPA', quality_ratio)
                return fallback(orig)
    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    except Exception:
        print('no contours')
        #cv2.imshow("Original", imutils.resize(orig, height=650))
        #cv2.waitKey(0)
        return None
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    t = threshold_local(warped, 11, offset=8, method="mean")
    #t = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    warped = (warped > t).astype("uint8") * 255

    cv2.imshow("Original", imutils.resize(gray_blurred, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.imshow("Edged", imutils.resize(edged, height=650))
    cv2.waitKey(0)
    img = Image.fromarray(warped, "L")
    img.save("image_to_send_to_ocr.jpg", "JPEG")
    with open('image_to_send_to_ocr.jpg', 'rb') as f:
        bc = f.read()
    dat = {'file': bc}
    #return dat
    return contour_area

def fallback(orig):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 17)
    cv2.imshow("orig", imutils.resize(t, height=650))
    cv2.waitKey(0)
    return t
    #t = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)