import argparse

# construct the argument parse and parse the arguments
import imutils
import numpy as np

import cv2
import pytesseract
from imutils import paths

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())

lower_blue = np.array([170, 100, 0], dtype ="uint8")
upper_blue = np.array([255, 150, 50], dtype = "uint8")

lower_orange = np.array([0, 90, 200], dtype = "uint8")
upper_orange = np.array([70, 160, 255], dtype = "uint8")
player_list = []

# prints out an array of the players on each team
def read_scoreboard(scoreboard):
    # cv2.imshow("scoreboard", scoreboard)
    # cv2.waitKey(0)
    players = []
    orange_players = read_colormasked_areas_scoreboard(scoreboard, lower_orange, upper_orange)
    blue_players = read_colormasked_areas_scoreboard(scoreboard, lower_blue, upper_blue)

    players.extend(orange_players)
    players.extend(blue_players)
    return players


# checks if scoreboard is present in the frame
def is_scoreboard(frame):
    return read_colormasked_areas_scoreboard_check(frame, lower_orange, upper_orange) \
           and read_colormasked_areas_scoreboard_check(frame, lower_blue, upper_blue)


# do the image processing
def process_scoreboard_roi(input_image, x, y, w):
    color_roi = input_image[y:round(y+w/4.8), x:x+w]
    bw_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(bw_roi, 120, 255, cv2.THRESH_BINARY_INV)
    thresh_roi = thresh[0:round(w/4.8), 200:650]
    # cv2.imshow("thresh_roi", thresh_roi)
    # cv2.waitKey(0)
    config = ("-l eng --oem 1 --psm 6")
    names = pytesseract.image_to_string(thresh_roi, config=config)
    return [s for s in names.splitlines() if s]


# scan the colormasked areas. if read_names = true, run the image_to_string
def read_colormasked_areas_scoreboard(input_image, lower_color, upper_color):
    cnts = find_color_contours(input_image, lower_color, upper_color)

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # if we have a blue or orange bar more than 1000 pixels across, it's probably the scoreboard
        if w > 800:
            # img_box = cv2.rectangle(input_image, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
            # cv2.imshow("img_box", img_box)
            # cv2.waitKey(0)
            return process_scoreboard_roi(input_image, x, y, w)

    return []


# scan the colormasked areas. if read_names = true, run the image_to_string
# else just return True or False whether the frame is showing a scoreboard
def read_colormasked_areas_scoreboard_check(input_image, lower_color, upper_color):
    cnts = find_color_contours(input_image, lower_color, upper_color)

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # if we have a blue or orange bar more than 1000 pixels across, it's probably the scoreboard
        if w > 800:
            return True

    return False


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


# for imagePath in paths.list_images(args["images"]):
#     image = cv2.imread(imagePath)
#     image = cv2.resize(image, (1920,1080))
#
#     read_scoreboard(image)
#     if not player_list and is_scoreboard(image):
#         player_list = read_scoreboard(image)
#         print(player_list)
