import pyttsx3
import os
from tkinter import *
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import scipy
from scipy.io.wavfile import write
import speech_recognition as sr
fs=44100
duration=5
root = Tk()
name = StringVar()

def question_to_seq(question, vocab_to_int):
	question = clean_text(question)
	temp = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]
	return temp

pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]

def cb():
    
	answer_string = ""
        input_question = e1.get()
        input_question = question_to_seq(input_question, questions_vocab_to_int)
	batch_shell = np.zeros((batch_size, max_line_length))
        input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))
        batch_shell[0] = input_question    

        answer_logits = sess.run(inference_logits, {input_data: batch_shell, 
                                                keep_prob: 1.0})[0]

        for i in np.argmax(answer_logits, 1):
            if i != pad_a:
                answer_string += str(answers_int_to_vocab[i])+" "
        print (answer_string)

        #set bot_text=bot response
        bot_text=answer_string
        #put testing file code
        e2.config(text=bot_text)
        #e3.config()
        if flag==0:
            engine = pyttsx3.init()
            engine.say(bot_text)
            engine.runAndWait()
        else:
            import os 
            inp = bot_text
            words = "\""+inp+"\""
            os.system("curl -H 'Content-Type: application/json' -H 'Authorization: Bearer oauth_1CzAzOfnqBL5FEaqXYCraEAdY0f' 'https://avatar.lyrebird.ai/api/v0/generate' -d '{ \"text\": "+words+"}' > audio.wav")
            os.system("aplay audio.wav")

def cb1():
	name.set(user_input.get())
	#set bot_text=bot response
	bot_text="current hello"
	#put testing file code
	e2.config(text=bot_text)
	#e3.config()
	engine = pyttsx3.init()
	engine.say(bot_text)
	engine.runAndWait()

Label(root, text="(Human)").grid(row=9,column=1,padx=100)
Label(root, text="(Friendroid)").grid(row=9,column=2,padx=200)
Label(root, text="").grid(row=1,column=3)

e1 = Entry(root)
e2 = Label(root, text='',width=30)

e1.grid(row=8, column=1)
e2.grid(row=8, column=2)

user=0
try:
	os.system("sudo streamer -f jpeg -o 1.jpeg")
	os.system("sudo convert 1.jpeg 1.png")
	user = PhotoImage(file = "1.png").subsample(1)
except:
	user = PhotoImage(file = "1.png").subsample(1)
label1 = Label(image=user)
label1.grid(row = 0, column = 1, sticky=NW,pady=50,padx=100)
friendroid = PhotoImage(file = "2.png").subsample(2)
label1 = Label(image=friendroid)
label1.grid(row = 0, column = 2, sticky=NW,pady=50,padx=100)

Button(root, text='Enter', command=cb).grid(row=10, column=1, sticky=W, padx=240,pady=10)
Button(root, text='Record', command=cb1).grid(row=12, column=1, sticky=W, padx=240,pady=10)
e3 = Label(root, width=30,textvariable=name)
e3.grid(row=7, column=5)
mainloop( )
