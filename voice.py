
#Speech to text

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr
#Frequency set
fs=44100
duration=5
#Recording sound
myrecording=sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')

print("Speak Something")
sd.wait()
print("Recording Done")

#Convereting recorded sound to an Numpy array and storing sound in sound.wav file
scaled = np.int16(myrecording/np.max(np.abs(myrecording)) * 32767)
write('sound.wav', 44100, scaled)
#Recognizing sound

to_speak=""
print("Recognizing Sound")
r = sr.Recognizer()
with sr.WavFile("sound.wav") as source:              
	audio = r.record(source)                        
try:
	to_speak=r.recognize_google(audio)
	print("Recorded Audio: " + r.recognize_google(audio))   
except LookupError:                                 
	pass

#Speak what was said
sd.play(myrecording, fs)

#Text to speech withoud personalized voice
import pyttsx3
engine = pyttsx3.init()
engine.say(to_speak)
engine.runAndWait()
'''
#MFCC feature extraction of sound in sound.wav
import speech_recognition as sr
import librosa
import IPython.display as ipd
data, fs= librosa.load('sound.wav')
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40).T,axis=0) 

#MFCC extraction(another method) and plotting
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import tkinter
(rate,sig) = wav.read("sound.wav")
mfcc_feat = mfcc(sig,rate)
plt.plot(mfcc_feat)
plt.title('MFCC feature plot')
plt.show()
plt.plot(mfccs)
plt.title('MFCC feature plot')
plt.show()
'''
