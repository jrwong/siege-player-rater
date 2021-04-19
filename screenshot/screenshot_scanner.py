from imutils import paths
import numpy as np
import argparse
import imutils
import pytesseract
import cv2
import datetime
from KillfeedEvent import KillfeedEvent
from PlayerInKillfeed import PlayerInKillfeed
from RoundContext import RoundContext
from ScoreTimeReadout import ScoreTimeReadout
import sys
import scoreboard_scanner
from fuzzywuzzy import process
from skimage.metrics import structural_similarity as ssim
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
# ap.add_argument("-s", "--settings", help="path to settings icon")
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
defuser_location_map = {30: 1, 80: 2, 130: 3, 180: 4, 230: 5}

time_config = ("-l eng -c tessedit_char_whitelist=0123456789: --oem 1 --psm 7")

player_list = []

# this represents the readout from the scoreboard
def scoreboard_readout(time, orange_score, blue_score):
    d = dict();
    d['time']   = time
    d['orange_score']  = orange_score
    d['blue_score']  = blue_score
    return d


# isolate scoreboard from screenshot and return: Time in round, Match Score
def read_round_time(input):
    isolate_scoreboard_start = datetime.datetime.now()

    #Isolate scoreboard location on a 1080p pic
    clock = input[70:120, 920:1000]

    #greyscale
    # TODO: address red flashing numbers
    roi_gray = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)
    (T, thresh) = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("roi_gray", roi_gray)
    # cv2.waitKey(0)

    time = read_text(clock, time_config, 180, cv2.THRESH_BINARY_INV)

    if not validate_timestamp(time):
        erosion = cv2.erode(thresh, kernel_1, iterations=1)
        contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # ensure at least some circles were found
        if contours is not None and hierarchy is not None:
            hierarchy = hierarchy[0]
            # loop over the (x, y) coordinates and radius of the circles
            for component in zip(contours, hierarchy):

                cnt = component[0]
                hr = component[1]
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

    return ScoreTimeReadout(time, "", "")


def read_round_score(frame):
    scoreboard = frame[50:150, 800:1120]
    config = ("-l eng -c tessedit_char_whitelist=0123456789 --oem 1 --psm 10")
    orange_score = read_colormasked_areas(scoreboard, lower_orange, upper_orange, config, "orange score: ")
    blue_score = read_colormasked_areas(scoreboard, lower_blue, upper_blue, config, "blue score: ")
    return {'orange': orange_score, 'blue': blue_score}



# scan in numbers from scoreboard
# TODO: THIS NEEDS TO BE FIXED UP
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
            color_img = cv2.adaptiveThreshold(color_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2)
            color_text = pytesseract.image_to_string(color_img, config=config)
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

            if 'has found the bomb' not in name:
                validated_name = validate_player(name, player_list)

                if validated_name != '':
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
        # print("best match for {}: {} with score {}", name, best_match[0], best_match[1])
        return best_match[0]
    else:
        print("player name {} badly missed the matching with a best match of {} and score {}", name, best_match[0], best_match[1])
        return ''


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
    # TODO: address Self destructs
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
    name_blocks = 0

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # if we have at least 3 blocks of blue or orange that have the proper w/h ratio
        # it's probably the round start
        if w*h > 5000 and w/h > 5:
            # cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            name_block = input_image[y:y+h, x:x+w]
            op_name_block = input_image[y-70: y-5, x+75: x+300]

            # config = ("-l eng --oem 1 --psm 6")
            # op_name_string = read_text(op_name_block, config, 170, cv2.THRESH_BINARY_INV)
            # print('op name ' + op_name_string)
            # # find_icon(op_icon, is_defense)
            # player_name_string = read_text(name_block, config, 170, cv2.THRESH_BINARY_INV)
            # print('player name is ' + player_name_string)
            name_blocks += 1
            if name_blocks > 4:
                return True

    return False


def check_for_settings_icon(input_image, reference_image):
    settings_section = input_image[30:180, 1700:1870]
    left_result = cv2.matchTemplate(settings_section, reference_image, cv2.TM_CCOEFF)
    (_, left_score, _, _) = cv2.minMaxLoc(left_result)
    # from testing, seems like the limit is about 10 million with 15 million being ideal. will modify later
    return left_score > 10000000


def check_for_infinity_symbol(input_image, reference_image):
    infinity_section = input_image[60:110, 900:1020]
    left_result = cv2.matchTemplate(infinity_section, reference_image, cv2.TM_CCOEFF)
    (_, left_score, _, _) = cv2.minMaxLoc(left_result)
    # from testing, seems like the limit is about 10 million with 15 million being ideal. will modify later
    return left_score > 10000000


# checks to indicate round start
# 1. check for settings icon in the proper location
# 2. check for time to be infinity symbol
# 3. check for 3 large enough blue or orange patches in the middle of screen to indicate round start
def is_round_start_screen(frame, settings_icon, infinity_symbol):
    if check_for_settings_icon(frame, settings_icon) and check_for_infinity_symbol(frame, infinity_symbol):
        usernames_section = frame[740:890, 90:1830]
        return round_start_color_check(usernames_section, lower_orange, upper_orange) \
                   or round_start_color_check(usernames_section, lower_blue, upper_blue)

    return False


# Checks to see if current player is on attack or defense
def defender_check(frame, defender_symbol):
    left_symbol = frame[65: 115, 805: 850]
    left_symbol = cv2.cvtColor(left_symbol, cv2.COLOR_BGR2GRAY)
    left_symbol = cv2.threshold(left_symbol, 180, 255, cv2.THRESH_BINARY_INV)[1]

    left_result = cv2.matchTemplate(defender_symbol, left_symbol, cv2.TM_CCOEFF)
    (_, left_score, _, _) = cv2.minMaxLoc(left_result)
    if left_score > 10000000:
        return True

    return False


# given frame, find operator lineup for current player and fetch location of defuser icon
def find_defuser(defuser_icon, frame):
    operator_lineup = frame[70:120, 550:800]
    gray = cv2.cvtColor(operator_lineup, cv2.COLOR_BGR2GRAY)
    found = None
    (tH, tW) = defuser_icon.shape[:2]
    # loop over the scales of the image
    for scale in np.linspace(1.0, 5.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, defuser_icon, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized
        # if args.get("visualize", False):
        #     # draw a bounding box around the detected region
        #     clone = np.dstack([edged, edged, edged])
        #     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
        #                   (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        #     cv2.imshow("Visualize", clone)
        #     cv2.waitKey(0)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    defuser_location_map = {30: 1, 80: 2, 130: 3, 180: 4, 230: 5}
    location = min(defuser_location_map, key=lambda x: abs(x-startX))
    print("op who has the bomb: " + str(defuser_location_map[location]))
    # draw a bounding box around the detected result and display the image
    # cv2.rectangle(operator_lineup, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # cv2.imshow("Image", operator_lineup)
    # cv2.waitKey(0)


# reads image using image_to_string with the given config string,
# gray_thresh_lower_bound being the lower limit of filter and thresh_type being the thresh type
def read_text(image_text, config, gray_thresh_lower_bound, thresh_type):
    gray_text = cv2.cvtColor(image_text, cv2.COLOR_BGR2GRAY)
    thresh_text = cv2.threshold(gray_text, gray_thresh_lower_bound, 255, thresh_type)[1]
    text = pytesseract.image_to_string(thresh_text, config=config)
    return text


def load_round_context(frame, defuser_canny, defender_symbol):
    is_defense = defender_check(frame, defender_symbol)
    round_score = read_round_score(frame)
    if is_defense:
        objective_location = frame[870:930, 850:1120]
        # cv2.imshow("objective_location", objective_location)
        # cv2.waitKey(0)
        config = ("-l eng --oem 1 --psm 6")
        location_text = read_text(objective_location, config, 170, cv2.THRESH_BINARY_INV)
        print("location is " + location_text)
    else:
        find_defuser(defuser_canny, frame)
    return RoundContext(round_score, is_defense)


# checks that timestamp read from image is the correct format (e.g. 0:45)
def validate_timestamp(time):
    return re.search("[0-9]:[0-9]{2}", time)


#
# for refPath in paths.list_images(args["ref"]):
#     # load image, resize, and convert to grayscale
#     if "defender_symbol" in refPath:
#         ref = cv2.imread(refPath)
#         ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]
#
#         defender_symbol = thresh
#
#     if "defuser_icon" in refPath:
#         ref = cv2.imread(refPath)
#         ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
#         ref = cv2.Canny(ref, 50, 200)
#         (tH, tW) = ref.shape[:2]
#         defuser_canny = ref
#
#     if "settings_icon" in refPath:
#         settings_icon = cv2.imread(refPath)
#
# try:
#     defender_symbol
#     defuser_canny
#     settings_icon
# except NameError:
#     print("missing key templates")
#     sys.exit(0)
#
#
# for imagePath in paths.list_images(args["images"]):
#     # load image, resize, and convert to grayscale
#     image = cv2.imread(imagePath)
#     image = cv2.resize(image, (1920, 1080))
#     # if not player_list and scoreboard_scanner.is_scoreboard(image):
#     #     player_list = scoreboard_scanner.read_scoreboard(image)
#     #     print(player_list)
#     # check_character_select(image, settings_icon)
#     find_defuser(defuser_canny, image)
#     if is_round_start_screen(image, settings_icon):
#         round_context = load_round_context(image)
#         print("round start")
#
#     if check_kill_feed(image):
#         kfes = read_kill_feed(image, [])
#         scoreboard = read_round_time(image)
#         for kf in kfes:
#             kf.scoreboard_readout = scoreboard
#             print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
#                   kf.death.desc + "team at " + kf.scoreboard_readout.time + ", with the score of blue:" +
#                   kf.scoreboard_readout.blue_score + " vs orange:" + kf.scoreboard_readout.orange_score)
#
#         # cv2.imshow(imagePath, image)
#         # cv2.waitKey(0)
