import sys
import os
from pydub import AudioSegment



'''
This file is for converting the mp3 files in common voice to wav format


'''

path = sys.argv[1]

#Change working directory
os.chdir(path)

audio_files = os.listdir()


for file in audio_files:
    name, ext = os.path.splitext(file)
    if ext == ".mp3":
       mp3_sound = AudioSegment.from_mp3(file)
       #rename them using the old name + ".wav"
       mp3_sound.export("{0}.wav".format(name), format="wav")