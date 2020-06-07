import streamlink
from subprocess import Popen
from time import sleep

# get the URL of .m3u8 file that represents the stream
stream_url = streamlink.streams('https://www.twitch.tv/breccoli')['best'].url
print(stream_url)

# now we start a new subprocess that runs ffmpeg and downloads the stream
ffmpeg_process = Popen(["ffmpeg", "-i", stream_url, "-c", "copy", 'stream.mkv'])

# we wait 60 seconds
sleep(120)

# terminate the process, we now have ~1 minute video of the stream
ffmpeg_process.kill()
