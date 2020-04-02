import unittest
import sys
sys.path.append('.')
from screenshot.screenshot_scanner import read_kill_feed
from screenshot.KillfeedEvent import KillfeedEvent
from screenshot.PlayerInKillfeed import PlayerInKillfeed
from screenshot.ScoreboardReadout import ScoreboardReadout


class TestScreenshotScanner(unittest.TestCase):

    def test_read_kill_feed(self):
        image = cv2.imread("bak/L33TBBQ_Rae_piste_142.jpg")
        image = cv2.resize(image, (1920,1080))
        kfes = read_kill_feed(image)
        self.assertEqual('foo'.upper(), 'FOO')
        for kf in kfes:
            kf['scoreboard_readout'] = scoreboard
            print(kf['kill']['name'] + " on the " + kf['kill']['desc'] + "team killed " + kf['death']['name'] + " on the " +
                  kf['death']['desc'] + "team at " + kf['scoreboard_readout']['time'] + ", with the score of blue:" +
                  kf['scoreboard_readout']['blue_score'] + " vs orange:" + kf['scoreboard_readout']['orange_score'])

if __name__ == '__main__':
    unittest.main()