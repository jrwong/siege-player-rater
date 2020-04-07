import argparse

# construct the argument parse and parse the arguments
import imutils
import numpy as np

import cv2
import pytesseract
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

lower_blue = np.array([170, 100, 0], dtype ="uint8")
upper_blue = np.array([255, 150, 50], dtype = "uint8")

lower_orange = np.array([0, 90, 200], dtype = "uint8")
upper_orange = np.array([70, 160, 255], dtype = "uint8")

def read_scoreboard(scoreboard):
    orange_players = read_colormasked_areas(scoreboard, lower_orange, upper_orange)
    print(orange_players)
    blue_players = read_colormasked_areas(scoreboard, lower_blue, upper_blue)
    print(blue_players)


def read_colormasked_areas(input_image, lower_color, upper_color):
    userNames = []
    cnts = find_color_contours(input_image, lower_color, upper_color)
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 1000:
            cv2.rectangle(input_image, (x, y), (x+w, round(y+w/4.8)), (255, 0, 0), 2)
            color_roi = input_image[y:round(y+w/4.8), x:x+w]
            bw_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            (T, thresh) = cv2.threshold(bw_roi, 180, 255, cv2.THRESH_BINARY_INV)
            thresh_roi = thresh[0:round(w/4.8), 200:650]
            config = ("-l eng --oem 1 --psm 6")
            names = pytesseract.image_to_string(thresh_roi, config=config)
            userNames.extend([s for s in names.splitlines() if s])
            cv2.imshow("thresh_roi", thresh_roi)
            cv2.waitKey(0)

    return userNames

    # cv2.imshow("rectangles", input_image)
    # cv2.waitKey(0)


# finds contours for a given color. filter by aspect ratio and/or size
def find_color_contours(input_image, lower_color, upper_color):
    # find the colors within the specified boundaries and apply
    # the mask
    mask_color = cv2.inRange(input_image, lower_color, upper_color)
    # cv2.imshow("mask_color", mask_color)
    # cv2.waitKey(0)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(mask_color.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts


for imagePath in paths.list_images(args["images"]):
    # load image, resize, and convert to grayscale
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1920,1080))

    read_scoreboard(image)
