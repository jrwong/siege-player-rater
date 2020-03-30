# import the necessary packages
import screenshot_scanner
from imutils import paths
import argparse
import time
import cv2
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to the (optional) video file")
ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])

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

# keep looping over the frames
while True:
    # grab the current frame and then handle if the frame is returned
    # from either the 'VideoCapture' or 'VideoStream' object,
    # respectively
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the
    # video
    if frame is None:
        break

    count += 1

    if count % frame_skips == 0 and screenshot_scanner.check_kill_feed(frame):
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        killfeed_events = screenshot_scanner.read_kill_feed(frame)
        scoreboard = screenshot_scanner.read_scoreboard(frame, killfeed_events)
        for kf in killfeed_events:
            kf['scoreboard_readout'] = scoreboard
            print(kf['kill']['name'] + " on the " + kf['kill']['desc'] + "team killed " + kf['death']['name'] + " on the " +
                  kf['death']['desc'] + "team at " + kf['scoreboard_readout']['time'])

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
# if we are not using a video file, stop the video file stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera pointer
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()