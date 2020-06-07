# import the necessary packages
import scoreboard_scanner
import screenshot_scanner
from imutils import paths
import argparse
import cv2
import datetime
from queue import Queue
import time
from event_compiler import compile_player_scores
# from collections import defaultdict

# construct the argument parse and parse the arguments
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to the (optional) video file")
ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
args = vars(ap.parse_args())
#
# vs = cv2.VideoCapture(args["video"])

for refPath in paths.list_images(args["ref"]):
    # load image, resize, and convert to grayscale
    ref = cv2.imread(refPath)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]

    attackersymbol = thresh

# skip once every second
frame_skips = 30
count = 0

vid_start = datetime.datetime.now()
running_killfeed = []

players = []
player_dict = {}

video_getter = VideoGet(args["video"]).start()
cap = cv2.VideoCapture(args["video"])
cps = CountsPerSec().start()
# Initializing a queue
q = Queue(maxsize = 100000)
num_fetch_threads = 2


# keep looping over the frames
while True:
    # grab the current frame and then handle if the frame is returned
    # from either the 'VideoCapture' or 'VideoStream' object,
    # respectively
    # if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
    #     video_getter.stop()
    #     break
    # frame = video_getter.frame

    (grabbed, frame) = cap.read()
    if grabbed:
        cps.increment()
        count += 1
        if count % frame_skips == 0:
            q.put(frame)
            if q.full():
                print("queue full!")
                break

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    else:
        break
# # if we are not using a video file, stop the video file stream
# if not args.get("video", False):
#     vs.stop()
# # otherwise, release the camera pointer
# else:
#     vs.release()
queue_end = datetime.datetime.now()
queue_time = queue_end-vid_start
print("queue time" + str(queue_time.seconds))
print("size of queue " + str(q.qsize()))
print("frame count" + str(count))

for curr_frame in list(q.queue):

    # TODO: implement queue for processing frames of interest

        # if the scoreboard is showing in the frame and the player list isn't populated, run scoreboard reader
        # if not players and scoreboard_scanner.is_scoreboard(frame):
        #     players = scoreboard_scanner.read_scoreboard(frame)
        #     print(players)
    curr_frame = cv2.resize(curr_frame, (1920,1080))

    round_num = 0
    if screenshot_scanner.is_round_start_screen(curr_frame):
        print("round start")

    if screenshot_scanner.check_kill_feed(curr_frame):
        killfeed_events = screenshot_scanner.read_kill_feed(curr_frame, players)
        scoreboard = screenshot_scanner.read_scoreboard(curr_frame)

        for kf in killfeed_events:
            kf.scoreboard_readout = scoreboard
            if kf not in running_killfeed:
                running_killfeed.append(kf)
                print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
                      kf.death.desc + "team at " + kf.scoreboard_readout.time)


compile_player_scores(running_killfeed, player_dict)

vid_end = datetime.datetime.now()
process_time = vid_end-vid_start;
print("process time" + str(process_time.seconds))
# close all windows
cv2.destroyAllWindows()