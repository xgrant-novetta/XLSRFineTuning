import sys
import os
import librosa
import soundfile as sf



'''
This file is for resampling the wav files to 16khz


'''

path = sys.argv[1]

#Change working directory
os.chdir(path)

audio_files = os.listdir()


for file in audio_files:
    y, sr = librosa.load(file)
    y_16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write(file, y_16, 16000)
    