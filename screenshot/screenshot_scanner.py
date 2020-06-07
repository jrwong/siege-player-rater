from imutils import paths
import numpy as np
import argparse
import imutils
import pytesseract
import cv2
import datetime
from KillfeedEvent import KillfeedEvent
from PlayerInKillfeed import PlayerInKillfeed
from ScoreTimeReadout import ScoreTimeReadout
from fuzzywuzzy import process
import re

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
#                 help="minimum probability required to inspect a region")
# ap.add_argument("-w", "--width", type=int, default=320,
#                 help="nearest multiple of 32 for resized width")
# ap.add_argument("-e", "--height", type=int, default=320,
#                 help="nearest multiple of 32 for resized height")
# ap.add_argument("-p", "--padding", type=float, default=0.0,
#                 help="amount of padding to add to each border of ROI")
# ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
# args = vars(ap.parse_args())

#Initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
kernel = np.ones((5,5),np.uint8)
kernel_1 = np.ones((2,2),np.uint8)

lower_blue = np.array([170, 100, 0], dtype ="uint8")
upper_blue = np.array([255, 150, 50], dtype = "uint8")

lower_orange = np.array([0, 90, 180], dtype = "uint8")
upper_orange = np.array([70, 160, 255], dtype = "uint8")

player_list = []

# this represents the readout from the scoreboard
def scoreboard_readout(time, orange_score, blue_score):
    d = dict();
    d['time']   = time
    d['orange_score']  = orange_score
    d['blue_score']  = blue_score
    return d


# isolate scoreboard from screenshot and return: Time in round, Match Score
def read_scoreboard(input):
    isolate_scoreboard_start = datetime.datetime.now()

    #Isolate scoreboard location on a 1080p pic
    clock = input[70:120, 920:1000]
    scoreboard = input[50:150, 800:1120]
    # cv2.imshow("roi", scoreboard)
    # cv2.waitKey(0)

    # left_symbol = input[65: 120, 805: 850]
    # left_symbol = cv2.cvtColor(left_symbol, cv2.COLOR_BGR2GRAY)
    # left_symbol = cv2.threshold(left_symbol, 180, 255, cv2.THRESH_BINARY_INV)[1]
    # # cv2.imshow("left_symbol", left_symbol)
    # # cv2.waitKey(0)
    # left_result = cv2.matchTemplate(attackersymbol, left_symbol, cv2.TM_CCOEFF)
    # (_, left_score, _, _) = cv2.minMaxLoc(left_result)
    # # print("left symbol result: " + str(left_score))
    #
    # right_symbol = input[60: 120, 1070: 1120]
    # right_symbol = cv2.cvtColor(right_symbol, cv2.COLOR_BGR2GRAY)
    # right_symbol = cv2.threshold(right_symbol, 180, 255, cv2.THRESH_BINARY_INV)[1]
    # right_result = cv2.matchTemplate(attackersymbol, right_symbol, cv2.TM_CCOEFF)
    # # cv2.imshow("right_symbol", right_symbol)
    # # cv2.waitKey(0)
    # (_, right_score, _, _) = cv2.minMaxLoc(right_result)
    # # print("right symbol result: " + str(right_score))

    #greyscale
    # TODO: address red flashing numbers
    roi_gray = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("roi_gray", roi_gray)
    # cv2.waitKey(0)

    config = ("-l eng -c tessedit_char_whitelist=0123456789: --oem 1 --psm 7")

    pytess_start = datetime.datetime.now()
    time = pytesseract.image_to_string(thresh, config=config)

    if not validate_timestamp(time):
        erosion = cv2.erode(thresh, kernel_1, iterations=1)
        contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        # ensure at least some circles were found
        if contours is not None:

            # loop over the (x, y) coordinates and radius of the circles
            for component in zip(contours, hierarchy):

                cnt = component[0]
                hr = component[1]
                # rect = cv2.minAreaRect(cnt)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(erosion,[box],0,(0,0,255),2)
                x,y,w,h = cv2.boundingRect(cnt)

                if 0 < y < 7:
                    cv2.drawContours(erosion,[cnt],0,(0,255,0),6)
                    # open contour
                    if hr[2] < 0:
                        perimeter = cv2.arcLength(cnt, False)
                        seconds_left = round(perimeter/260 * 45)
                        time = "{} seconds left to defuse".format(seconds_left)
                    # closed contour
                    else:
                        time = "45 seconds left to defuse"

    # if still no time found, it's the end of the round
    if not time:
        time = "end of round"

    pytess_end = datetime.datetime.now()
    pytess_diff= pytess_end-pytess_start;
    # print("time is " + time)
    # print("pytess time " + str(pytess_diff.microseconds))

    # config = ("-l eng -c tessedit_char_whitelist=012345 --oem 1 --psm 10")
    # orange_score = read_colormasked_areas(scoreboard, lower_orange, upper_orange, config, "orange score: ")
    # blue_score = read_colormasked_areas(scoreboard, lower_blue, upper_blue, config, "blue score: ")

    isolate_scoreboard_end = datetime.datetime.now()
    total_func_time = isolate_scoreboard_end - isolate_scoreboard_start;
    # print("total func time " + str(total_func_time.microseconds))

    return ScoreTimeReadout(time, "", "")


# scan in numbers from scoreboard
def read_colormasked_areas(input_image, lower_color, upper_color, config, desc):

    cnts = find_color_contours(input_image, lower_color, upper_color)

    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since score will be a fixed size of about 25 x 35, we'll set the area at about 300 to be safe
        if w*h > 300:
            color_img = input_image[y-10:y+h+10, x-10:x+w+10]
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            color_img = cv2.adaptiveThreshold(color_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            # cv2.imshow("orange_score_img adaptivethresh", color_img)
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
            # TODO: improve scoreboard reliability?
            if color_text == "":
                print("missed reading!!! for " + desc)
                cv2.imshow("scoreboard_number", input_image)
                cv2.waitKey(0)
                continue
            return color_text


# scan in numbers from killfeed.
# TODO: Will want to merge with read_colormasked_areas for readability at some point
def read_colormasked_areas_killfeed(input_image, lower_color, upper_color, config, desc):

    # increase killfeed size for readability
    (origH, origW) = input_image.shape[:2]
    # resize the width to be 1280 pixels and grab the new image dimensions
    r = 1280.0 / origW
    dim = (1280, int(origH * r))
    input_image = cv2.resize(input_image, dim)
    # cv2.imshow("input_image", input_image)
    # cv2.waitKey(0)

    cnts = find_color_contours(input_image, lower_color, upper_color)

    players = []

    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since score will be a fixed size of about 25 x 35, we'll set the area at about 300 to be safe
        if w*h > 5000 and ar > 2:
            killfeed_roi = input_image[y:y+h, x+15:x+w-15]
            person1 = cv2.cvtColor(killfeed_roi, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("b2gray", person1)
            # cv2.waitKey(0)
            (T, thresh) = cv2.threshold(person1, 180, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
            # cv2.imshow("thresh", thresh)
            # cv2.waitKey(0)

            name = pytesseract.image_to_string(thresh, config=config)
            validated_name = name
            # validated_name = validate_player(name, player_list)
            # print(desc + name)
            # print(desc + " x value: " + str(x) + ", y value: " + str(y))
            player = PlayerInKillfeed(desc, validated_name, x, y)
            # add player to the list of players currently in the killfeed
            players.append(player)

    return players


# validates player by checking against player list if it exists
def validate_player(name, player_list):
    if not player_list:
        return name

    best_match = process.extractOne(name, player_list)
    if best_match[1] > 80:
        print("best match for {}: {} with score {}", name, best_match[0], best_match[1])
        return best_match[0]
    else:
        print("player name {} badly missed the matching with a best match of {} and score {}", name, best_match[0], best_match[1])
        return name


# finds contours for a given color. filter by aspect ratio and/or size
def find_color_contours(input_image, lower_color, upper_color):
    # find the colors within the specified boundaries and apply
    # the mask
    mask_color = cv2.inRange(input_image, lower_color, upper_color)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(mask_color.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts


def read_kill_feed(input, players):
    global player_list
    player_list = players
    kill_feed = input[250:450, 1420:1920]
    config = ("-l eng --oem 1 --psm 8")
    orange_players = read_colormasked_areas_killfeed(kill_feed, lower_orange, upper_orange, config, "orange ")
    blue_players = read_colormasked_areas_killfeed(kill_feed, lower_blue, upper_blue, config, "blue ")

    return sort_kill_feed(blue_players, orange_players)


# given two sets of players, orange and blue, sort them into who killed who
def sort_kill_feed(blue_players, orange_players):

    # run some validation here to check against known players from scoreboard reading
    players = blue_players
    players.extend(orange_players)

    players = sorted(players, key = lambda i: i.y)

    # loop through and check if previous y value is within range of the current player
    # if so, then create a killfeed_event
    # TODO: address Self destructs, "x found the bomb" lines
    killfeed_events = []
    prev_y_val = 0
    prev_player = PlayerInKillfeed(None, None, None, None)
    for player in players:
        if player.y in range(prev_y_val-10, prev_y_val+10) and prev_player.x and player.x:
            if prev_player.x < player.x:
                kf = KillfeedEvent(prev_player, player, "")
            else:
                kf = KillfeedEvent(player, prev_player, "")
            killfeed_events.append(kf)
        prev_player = player
        prev_y_val = player.y

    return killfeed_events


def check_kill_feed(input):

    kill_feed = input[250:450, 1420:1920]

    # increase killfeed size for readability
    (origH, origW) = kill_feed.shape[:2]
    # resize the width to be 1280 pixels and grab the new image dimensions
    r = 1280.0 / origW
    dim = (1280, int(origH * r))
    kill_feed = cv2.resize(kill_feed, dim)

    cnts = find_color_contours(kill_feed, lower_orange, upper_orange)
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # checking for a box with an area of 5k+ pixels and an aspect ratio of > 2:1
        if w*h > 5000 and ar > 2:
            return True

    cnts = find_color_contours(kill_feed, lower_blue, upper_blue)
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # checking for a box with an area of 5k+ pixels and an aspect ratio of > 2:1
        if w*h > 5000 and ar > 2:
            return True

    return False


# scan the colormasked areas. if read_names = true, run the image_to_string
# else just return True or False whether the frame is showing a scoreboard
def round_start_color_check(input_image, lower_color, upper_color):
    cnts = find_color_contours(input_image, lower_color, upper_color)
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # if we have a blue or orange bar more than 1000 pixels across, it's probably the scoreboard
        if w*h > 5000:
            cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.imshow("round_start_color_check", input_image)
            # cv2.waitKey(0)
            return True

    return False


# def check_character_select(input_image, reference_image):
#     settings_section = input_image[50:150, 1720:1920]
#     cv2.imshow("settings_section", settings_section)
#     cv2.waitKey(0)


# checks for large enough blue and orange patches in the middle of screen to indicate round start
def is_round_start_screen(frame):
    middle_section = frame[200:400, 750:1150]
    # cv2.imshow("middle_section", middle_section)
    # cv2.waitKey(0)
    return round_start_color_check(middle_section, lower_orange, upper_orange) \
           and round_start_color_check(middle_section, lower_blue, upper_blue)


# checks that timestamp read from image is the correct format (e.g. 0:45)
def validate_timestamp(time):
    return re.search("[0-9]:[0-9]{2}", time)
#
#
# for refPath in paths.list_images(args["ref"]):
#     # load image, resize, and convert to grayscale
#     ref = cv2.imread(refPath)
#     ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]
#
#     attackersymbol = thresh
#     # cv2.imshow(refPath, thresh)
#     # cv2.waitKey(0)
#
#
# for imagePath in paths.list_images(args["images"]):
#     # load image, resize, and convert to grayscale
#     image = cv2.imread(imagePath)
#     image = cv2.resize(image, (1920, 1080))
#     # check_character_select(image, image)
#
#
#     check_kill_feed(image)
#
#     kfes = read_kill_feed(image, [])
#     scoreboard = read_scoreboard(image)
#     for kf in kfes:
#         kf.scoreboard_readout = scoreboard
#         print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
#               kf.death.desc + "team at " + kf.scoreboard_readout.time + ", with the score of blue:" +
#               kf.scoreboard_readout.blue_score + " vs orange:" + kf.scoreboard_readout.orange_score)
#
#     cv2.imshow(imagePath, image)
#     cv2.waitKey(0)
