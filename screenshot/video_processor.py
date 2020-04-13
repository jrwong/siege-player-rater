# import the necessary packages
import scoreboard_scanner
import screenshot_scanner
from imutils import paths
import argparse
import cv2
import datetime

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
frame_skips = 60
count = 0

vid_start = datetime.datetime.now()
running_killfeed = []

players = []

video_getter = VideoGet(args["video"]).start()
cps = CountsPerSec().start()

# keep looping over the frames
while True:
    # grab the current frame and then handle if the frame is returned
    # from either the 'VideoCapture' or 'VideoStream' object,
    # respectively
    if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
        video_getter.stop()
        break
    frame = video_getter.frame

    count += 1

    cps.increment()

    if count % frame_skips == 0:
        # if the scoreboard is showing in the frame and the player list isn't populated, run scoreboard reader
        if not players and scoreboard_scanner.is_scoreboard(frame):
            players = scoreboard_scanner.read_scoreboard(frame)
            print(players)

        if screenshot_scanner.check_kill_feed(frame):
            killfeed_events = screenshot_scanner.read_kill_feed(frame, players)
            scoreboard = screenshot_scanner.read_scoreboard(frame)
            # TODO: filter out similar events from previous seconds
            for kf in killfeed_events:
                kf.scoreboard_readout = scoreboard
                if kf not in running_killfeed:
                    running_killfeed.append(kf)
                    print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
                          kf.death.desc + "team at " + kf.scoreboard_readout.time)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
# # if we are not using a video file, stop the video file stream
# if not args.get("video", False):
#     vs.stop()
# # otherwise, release the camera pointer
# else:
#     vs.release()

vid_end = datetime.datetime.now()
process_time = vid_end-vid_start;
print(str(process_time.seconds))
# close all windows
cv2.destroyAllWindows()