from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils


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


def find_best_ratio(photo_path, ratio_parameter: int, poly_dp_ratio=0.02):
    image = cv2.imread(photo_path)
    image = imutils.resize(image, height=ratio_parameter)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 7)
    edged = auto_canny(gray_blurred)
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]

    contour_area = 0
    # loop over the contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, poly_dp_ratio * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our business card
        if len(approx) == 4:
            contour_area = cv2.contourArea(c)
            break

    return contour_area


def show_segment(photo_path, ratio_parameter: int, poly_dp_ratio=0.02):
    global quality_ratio
    image = cv2.imread(photo_path)
    ratio = image.shape[0] / ratio_parameter
    orig = image.copy()
    image = imutils.resize(image, height=ratio_parameter)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 7)
    edged = auto_canny(gray_blurred)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    contour = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)[:]

    screen_cnt = False
    # loop over the contours
    for c in contour:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, poly_dp_ratio * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screen_cnt = approx
            contour_area = cv2.contourArea(c)
            quality_ratio = contour_area / (image.shape[0] * image.shape[1])
            break

    if screen_cnt is False or quality_ratio < 0.15:
        # print('trying to close contour')
        edged = cv2.Canny(gray_blurred, 10, 300)
        closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        contour = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        contour = sorted(contour, key=cv2.contourArea, reverse=True)[:]
        # loop over the contours
        for c in contour:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, poly_dp_ratio * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screen_cnt = approx
                contour_area = cv2.contourArea(c)
                quality_ratio = contour_area / (image.shape[0] * image.shape[1])
                # todo why do i need this below?
                if screen_cnt is False or quality_ratio < 0.15:
                    print('no accurate contour detected', quality_ratio)
                    return fallback(orig)
                break
    if screen_cnt is False or quality_ratio < 0.15:
        print('no accurate contour detected', quality_ratio)
        return fallback(orig)
    try:
        warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
    except Exception as e:
        print(e)
        return fallback(orig)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    t = threshold_local(warped, 11, offset=8, method="mean")
    warped = (warped > t).astype("uint8") * 255

    return warped


# if there's no accurate segment to show
def fallback(orig):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 17)
    return t
