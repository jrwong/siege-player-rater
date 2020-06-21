import customfilevideostream
from imutils.video import FPS
from imutils import paths
from customfilevideostream import CustomFileVideoStream
from imutils.video import FileVideoStream
import streamlink
import time
import cv2
from videostreamer import VideoStreamer
import scoreboard_scanner
import screenshot_scanner
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
ap.add_argument("-s", "--settings", help="path to settings icon")
args = vars(ap.parse_args())


for refPath in paths.list_images(args["ref"]):
    # load image, resize, and convert to grayscale
    ref = cv2.imread(refPath)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]

    attackersymbol = thresh

settings_icon = cv2.imread(args["settings"])

stream_url = streamlink.streams('https://clips.twitch.tv/TubularAbnegateKittenResidentSleeper')['best'].url
print(stream_url)

running_killfeed = []
sample_frequency = 10
players = []
player_dict = {}

print("[INFO] starting video file thread...")
# vs = CustomFileVideoStream(stream_url, sample_frequency).start()
# fvs = FileVideoStream(stream_url).start()
vs = VideoStreamer('https://www.twitch.tv/frostyjayy', queueSize=128, n_frame=30)
time.sleep(10.0)

# start the FPS timer
fps = FPS().start()

while True:
    if vs.more():
        frame = vs.read()
        curr_frame = cv2.resize(frame, (1920,1080))

        # if the scoreboard is showing in the frame and the player list isn't populated, run scoreboard reader
        if not players and scoreboard_scanner.is_scoreboard(curr_frame):
            players = scoreboard_scanner.read_scoreboard(curr_frame)
            print(players)

        elif screenshot_scanner.is_round_start_screen(curr_frame, settings_icon):
            print("round start")

        elif screenshot_scanner.check_kill_feed(curr_frame):
            killfeed_events = screenshot_scanner.read_kill_feed(curr_frame, players)
            scoreboard = screenshot_scanner.read_scoreboard(curr_frame)

            for kf in killfeed_events:
                kf.scoreboard_readout = scoreboard
                if kf not in running_killfeed:
                    running_killfeed.append(kf)
                    print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
                          kf.death.desc + "team at " + kf.scoreboard_readout.time)

        # display the size of the queue on the frame
        cv2.putText(frame, "Queue Size: {}".format(vs.Q.qsize()),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(100)
        fps.update()
    # wait a bit to fill up the queue
    else:
        time.sleep(10.0)
    #
    # else:
    #     stream_url = streamlink.streams('https://www.twitch.tv/easilyyr6')['best'].url
    #     print(stream_url)
    #     print("[INFO] starting video file thread again...")
    #     fvs = CustomFileVideoStream(stream_url, sample_frequency).start()
    #     time.sleep(4.0)

print("nothing left in the queue")
