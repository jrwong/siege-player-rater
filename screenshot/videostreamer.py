import ffmpeg
import streamlink
import numpy
from threading import Thread
import subprocess as sp
from queue import Queue

# mostly taken from https://github.com/DanielTea/rage-analytics/blob/master/engine/realtime_VideoStreamer.py


class VideoStreamer:
    def __init__(self, url: object, is_stream: object, queue_size: object = 128, resolution: object = '1080p60',
                 sample_freq: object = 0.5) -> object:
        self.stopped = False
        self.url = url
        self.res = resolution
        self.sample_freq = sample_freq
        self.is_stream = is_stream

        # initialize the queue used to store frames read from
        # the video stream
        self.Q = Queue(maxsize=queue_size)
        checkIfStreamsWorks = self.create_pipe()

        if checkIfStreamsWorks:
            self.start_buffer()

    def create_pipe(self):
        if self.is_stream:
            streamer_name = self.url.split("/")[3]


            try:
                streams = streamlink.streams(self.url)
            except streamlink.exceptions.NoPluginError:
                print("NO STREAM AVAILABLE for " + streamer_name)
                return False
            except:
                print("NO STREAM AVAILABLE no exception " + streamer_name)
                return False

            #print("available streams: "+ str(streams))

            # priority is highest resolution, and higher fps stream takes next
            resolutions = {'1080p60': {"byte_length": 1920, "byte_width": 1080}, '1080p': {"byte_length": 1920, "byte_width": 1080},
                           '720p60': {"byte_length": 1280, "byte_width": 720}, '720p': {"byte_length": 1280, "byte_width": 720},
                           '480p': {"byte_length": 854, "byte_width": 480}, '360p': {"byte_length": 640, "byte_width": 360}
                           }


            if self.res in streams:
                finalRes = self.res
            else:
                for key in resolutions:
                    if key != self.res and key in streams:
                        print("USED FALL BACK " + key)
                        finalRes = key
                        break
                else:
                    print("COULD NOT FIND STREAM " + streamer_name)
                    return False

            if 'p60' in finalRes:
                self.n_frame = round(60*self.sample_freq)
                print('60fps, sampling every ' + str(self.n_frame) + ' frames')
            else:
                self.n_frame = round(30*self.sample_freq)
                print('30fps, sampling every ' + str(self.n_frame) + ' frames')
            self.byte_length = resolutions[finalRes]["byte_length"]
            self.byte_width = resolutions[finalRes]["byte_width"]

            print("FINAL RES " + finalRes + " " + streamer_name)

            stream = streams[finalRes]
            stream_url = stream.url
        else:
            self.byte_length = 2560
            self.byte_width = 1440
            self.n_frame = round(60*self.sample_freq)
            stream_url = self.url


        self.pipe = sp.Popen(['ffmpeg', "-i", stream_url,
                              "-loglevel", "quiet",  # no text output
                              "-an",  # disable audio
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)
        return True

    def start_buffer(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update_buffer, args=())
        t.daemon = True
        t.start()
        return self

    def update_buffer(self):

        count_frame = 0

        while True:

            raw_image = self.pipe.stdout.read(
                self.byte_length * self.byte_width * 3)  # read length*width*3 bytes (= 1 frame)

            if count_frame % self.n_frame == 0:

                frame = numpy.fromstring(raw_image, dtype='uint8').reshape((self.byte_width, self.byte_length, 3))

                if not self.Q.full():
                    self.Q.put(frame)
                    count_frame += 1
                else:
                    count_frame += 1
                    continue
            else:
                count_frame += 1
                continue

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True