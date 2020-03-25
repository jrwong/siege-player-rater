from imutils import paths
import numpy as np
import argparse
import imutils
import pytesseract
import cv2
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
                help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
                help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

#Initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
kernel = np.ones((5,5),np.uint8)

lower_blue = np.array([170, 100, 0], dtype ="uint8")
upper_blue = np.array([255, 150, 50], dtype = "uint8")

lower_orange = np.array([0, 90, 200], dtype = "uint8")
upper_orange = np.array([70, 160, 255], dtype = "uint8")

# isolate scoreboard from screenshot and return: Time in round, Match Score
def read_scoreboard(input):
    isolate_scoreboard_start = datetime.datetime.now()

    #Isolate scoreboard location on a 1080p pic
    clock = input[70:120, 920:1000]
    scoreboard = input[50:150, 800:1120]
    # cv2.imshow("roi", scoreboard)
    # cv2.waitKey(0)

    #greyscale
    roi_gray = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.bitwise_not(roi_gray)

    config = ("-l eng -c tessedit_char_whitelist=0123456789: --oem 1 --psm 8")

    pytess_start = datetime.datetime.now()
    time = pytesseract.image_to_string(roi_gray, config=config)
    pytess_end = datetime.datetime.now()
    pytess_diff= pytess_end-pytess_start;
    print("time is " + time)
    # print("pytess time " + str(pytess_diff.microseconds))

    config = ("-l eng -c tessedit_char_whitelist=012345 --oem 1 --psm 10")
    read_colormasked_areas(scoreboard, lower_orange, upper_orange, config, "orange score: ")
    read_colormasked_areas(scoreboard, lower_blue, upper_blue, config, "blue score: ")

    isolate_scoreboard_end = datetime.datetime.now()
    total_func_time = isolate_scoreboard_end - isolate_scoreboard_start;
    print("total func time " + str(total_func_time.microseconds))


def read_colormasked_areas(input_image, lower_color, upper_color, config, desc):
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

    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since score will be a fixed size of about 25 x 35, we'll set the area at about 300 to be safe
        if w*h > 300:
            color_img = mask_color[y-10:y+h+10, x-10:x+w+10]
            color_img = cv2.adaptiveThreshold(color_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            # cv2.imshow("orange_score_img adaptivethresh", orange_score_img)
            # cv2.waitKey(0)
            color_img = cv2.dilate(color_img,kernel,iterations=1)
            color_img = cv2.GaussianBlur(color_img, (5, 5), 0)
            color_img = cv2.bitwise_not(color_img)
            # cv2.imshow("color_img final", color_img)
            # cv2.waitKey(0)
            pytess_start = datetime.datetime.now()
            color_text = pytesseract.image_to_string(color_img, config=config)
            pytess_end = datetime.datetime.now()
            pytess_diff= pytess_end-pytess_start;
            # print("color pytess time is " + str(pytess_diff.microseconds))
            print(desc + color_text)
            if (color_text == ""):
                print("missed reading!!! for " + desc)


def read_colormasked_areas_killfeed(input_image, lower_color, upper_color, config, desc):

    (origH, origW) = input_image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (1280, 640)
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    input_image = cv2.resize(input_image, (newW, newH))
    (H, W) = input_image.shape[:2]
    # cv2.imshow("input_image", input_image)
    # cv2.waitKey(0)

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

    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since score will be a fixed size of about 25 x 35, we'll set the area at about 300 to be safe
        if w*h > 5000 and ar > 2:
            input_image = input_image[y:y+h, x+15:x+w-15]
            person1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("b2gray", person1)
            # cv2.waitKey(0)
            (T, thresh) = cv2.threshold(person1, 180, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
            # cv2.imshow("thresh", thresh)
            # cv2.waitKey(0)

            text = pytesseract.image_to_string(thresh, config=config)
            print(desc + text)
            print(desc + " x value: " + str(x))

def read_kill_feed(input):
    kill_feed = input[250:450, 1520:1920]
    config = ("-l eng --oem 1 --psm 6")
    read_colormasked_areas_killfeed(kill_feed, lower_orange, upper_orange, config, "orange player: ")
    read_colormasked_areas_killfeed(kill_feed, lower_blue, upper_blue, config, "blue player: ")



for imagePath in paths.list_images(args["images"]):
    # load image, resize, and convert to grayscale
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1920,1080))
    # cv2.imshow(imagePath, image)
    # cv2.waitKey(0)
    read_kill_feed(image)
    read_scoreboard(image)
