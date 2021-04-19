from imutils.video import FPS
from imutils import paths
import time
import cv2
from videostreamer import VideoStreamer
import scoreboard_scanner
import screenshot_scanner
import argparse
from RoundContext import RoundContext


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--ref", required=True, help="path to reference images directory")
ap.add_argument("-s", "--settings", help="path to settings icon")
ap.add_argument("-so", "--source", help="path to video file if running video")
ap.add_argument("-db", "--debug", help="debug mode")
ap.add_argument("-dbo", "--debugOutput", help="kill feed screen shot output file")
args = vars(ap.parse_args())

# preload basic images for template comparison
for refPath in paths.list_images(args["ref"]):
    # TODO: move this load images to its own method, add filename check
    # load image, resize, and convert to grayscale
    ref = cv2.imread(refPath)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]

    attackersymbol = thresh

    # load image, resize, and convert to grayscale
    if "defender_symbol" in refPath:
        ref = cv2.imread(refPath)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(ref, 180, 255, cv2.THRESH_BINARY_INV)[1]

        defender_symbol = thresh

    if "defuser_icon" in refPath:
        ref = cv2.imread(refPath)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.Canny(ref, 50, 200)
        (tH, tW) = ref.shape[:2]
        defuser_canny = ref

    if "settings_icon" in refPath:
        settings_icon = cv2.imread(refPath)

    if "infinity_symbol" in refPath:
        infinity_symbol = cv2.imread(refPath)


if args["source"]:
    is_stream = False
    source_url = args["source"]
else:
    is_stream = True
    source_url = 'https://www.twitch.tv/easilyyr6'

debug_output_location = ""
if args["debugOutput"]:
    debug_output_location = args["debugOutput"]

running_killfeed = []
sample_frequency = 10
players = []
player_dict = {}

print("[INFO] starting video file thread...")
# TODO: create properties file for configs
vs = VideoStreamer(url=source_url, is_stream=is_stream, queue_size=128, sample_freq=0.5)
time.sleep(10.0)

# start the FPS timer
fps = FPS().start()

round_context = RoundContext(None, None)

while True:
    if vs.more():
        frame = vs.read()
        curr_frame = cv2.resize(frame, (1920, 1080))

        # if the scoreboard is showing in the frame and the player list isn't populated, run scoreboard reader
        if not players and scoreboard_scanner.is_scoreboard(curr_frame):
            players = scoreboard_scanner.read_scoreboard(curr_frame)
            print(players)

        # TODO: fix is_round_start_screen for current season
        elif screenshot_scanner.is_round_start_screen(curr_frame, settings_icon, infinity_symbol):
            old_round_number = 0
            if round_context.round_score is not None:
                old_round_number = round_context.round_score['orange'] + round_context.round_score['blue']

            round_context = screenshot_scanner.load_round_context(curr_frame, defuser_canny, defender_symbol)
            new_round_number = round_context.round_score['orange'] + round_context.round_score['blue']
            if old_round_number is None or new_round_number != old_round_number:
                print('round ' + new_round_number + ' start')

        elif screenshot_scanner.check_kill_feed(curr_frame):
            killfeed_events = screenshot_scanner.read_kill_feed(curr_frame, players)
            scoreboard = screenshot_scanner.read_round_time(curr_frame)

            for kf in killfeed_events:
                kf.scoreboard_readout = scoreboard
                if kf not in running_killfeed:
                    running_killfeed.append(kf)
                    print(kf.kill.name + " on the " + kf.kill.desc + "team killed " + kf.death.name + " on the " +
                          kf.death.desc + "team at " + kf.scoreboard_readout.time)
                    if args["debug"] == "true":
                        file_name = debug_output_location + kf.kill.name + "_" + kf.death.name + "_" + kf.scoreboard_readout.time + ".png"
                        cv2.imwrite(file_name, curr_frame)

        # display the size of the queue on the frame
        cv2.putText(frame, "Queue Size: {}".format(vs.Q.qsize()),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(50)
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
