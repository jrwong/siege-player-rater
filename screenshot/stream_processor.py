import customfilevideostream
from imutils.video import FPS
from imutils import paths
from customfilevideostream import CustomFileVideoStream
import streamlink
import time
import cv2
import scoreboard_scanner
import screenshot_scanner
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
args = vars(ap.parse_args())


for refPath in paths.list_images(args["ref"]):
    # load image, resize, and convert to grayscale
    ref = cv2.imread(refPath)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]

    attackersymbol = thresh

stream_url = streamlink.streams('https://www.twitch.tv/leongids')['best'].url
print(stream_url)

running_killfeed = []
sample_frequency = 10
players = []
player_dict = {}

print("[INFO] starting video file thread...")
fvs = CustomFileVideoStream(stream_url, sample_frequency).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

while fvs.more():
    frame = fvs.read()
    curr_frame = cv2.resize(frame, (1920,1080))

    # if the scoreboard is showing in the frame and the player list isn't populated, run scoreboard reader
    if not players and scoreboard_scanner.is_scoreboard(frame):
        players = scoreboard_scanner.read_scoreboard(frame)
        print(players)

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

    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(200)
    fps.update()

